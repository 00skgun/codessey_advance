"""Rebuild all evidence, the Korean README and an executable notebook."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.pipeline import DataAnalyzer


def markdown_table(frame):
    """Avoid an extra tabulate dependency."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |\n"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |\n"
    rows = frame.to_numpy().astype(str)
    return header + separator + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def create_figures(analyzer, rfm):
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
    figures = {}
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log10(df.price), bins=65, color="#327da8", edgecolor="white")
    ax.set(title="Unit price distribution after imputation", xlabel="log10(unit price in GBP)", ylabel="Order lines")
    figures["01_histogram"] = fig

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([df.price.to_numpy(), df.price_clipped.to_numpy()],
               tick_labels=["Before IQR clipping", "After IQR clipping"],
               flierprops={"markersize": 2, "alpha": 0.15})
    ax.set_yscale("log")
    ax.set(title="IQR treatment changes the display, not RFM accounting", xlabel="EDA treatment", ylabel="Unit price (GBP, log scale)")
    figures["02_boxplot"] = fig

    order = ["VIP", "Loyal", "New", "Potential", "Churned"]
    counts = rfm.segment.value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color=["#d29c3b", "#327da8", "#52a88c", "#959cc3", "#c86b70"])
    for position, value in enumerate(counts):
        ax.text(position, value + max(counts) * 0.015, f"{value:,}", ha="center")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set(title="Customer count by RFM segment", xlabel="Segment", ylabel="Customers")
    figures["03_bar"] = fig

    cols = ["price", "quantity", "amount", "name_length", "image_mean", "image_std"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set(title="Pearson correlation; image fixtures are synthetic", xlabel="Feature", ylabel="Feature")
    figures["04_heatmap"] = fig

    purchase = analyzer.valid_transactions()
    sample = purchase.sample(n=min(15000, len(purchase)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(np.log10(sample.quantity), np.log10(sample.amount), alpha=0.15, s=10, color="#327da8", rasterized=True)
    ax.set(title="Quantity and line spend (15,000-line fixed sample)", xlabel="log10(quantity)", ylabel="log10(gross line spend in GBP)")
    figures["05_scatter"] = fig

    monthly = purchase.groupby(purchase.order_date.dt.to_period("M")).amount.sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index.astype(str), monthly.values / 1000, marker="o", color="#327da8", linewidth=2)
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="Monthly identified gross purchases; final month is partial", xlabel="Order month (December 2011: days 1-9)", ylabel="Gross purchases (GBP thousands)")
    figures["06_line"] = fig
    for fig in figures.values():
        fig.tight_layout()
    return figures


def analyze():
    analyzer = DataAnalyzer(ROOT / "data/raw/transactions.csv")
    analyzer.load_data()
    missing_before = analyzer.df.isna().sum()
    analyzer.handle_missing_values(strategy="median", group_col="product_id")
    with np.load(ROOT / "data/image_fixtures.npz", allow_pickle=False) as images:
        analyzer.engineer_features(images["product_ids"], images["images"])
    analyzer.detect_outliers("price", threshold=1.5, treatment="clip")
    rfm = analyzer.calculate_rfm()
    summary = rfm.groupby("segment").agg(
        customers=("monetary", "size"), mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"),
        gross_purchases=("monetary", "sum"))
    summary["customer_share_pct"] = summary.customers / len(rfm) * 100
    summary["spend_share_pct"] = summary.gross_purchases / summary.gross_purchases.sum() * 100
    return analyzer, rfm, summary, missing_before


def interpretations(a, rfm, summary):
    vip, churn, new = (summary.loc[name] for name in ["VIP", "Churned", "New"])
    return [
        f"**VIP 유지** — (근거) VIP {int(vip.customers):,}명은 고객의 {vip.customer_share_pct:.1f}%이지만 식별 고객 총 구매액의 {vip.spend_share_pct:.1f}%를 차지한다. "
        "(실행) VIP 고객을 대상으로 우선 재입고 알림과 전담 응대 혜택을 제공하여 재구매 유지율 개선을 목표로 한다. "
        "(검증) 실제 순이익·반품·캠페인 노출 데이터가 필요하며, 무작위 대조군 대비 60일 재구매율과 공헌이익이 개선되지 않으면 전략을 기각한다.",
        f"**휴면 위험 고객 복귀** — (근거) Churned {int(churn.customers):,}명({churn.customer_share_pct:.1f}%)의 평균 미구매 기간은 {churn.mean_recency:.1f}일이다. "
        "(실행) Churned 중 과거 구매액이 높은 고객에게 개인화 재입고 안내를 보내고, 소규모 쿠폰 실험으로 30일 복귀율 개선을 목표로 한다. "
        "(검증) Churned는 상대적 최근성 점수에 붙인 이름이며 실제 이탈 확정이 아니다. 수신 동의·시즌성·할인 비용 데이터를 확보하고 대조군 대비 증분 이익으로 검증한다.",
        f"**첫 관측 구매 고객의 두 번째 주문** — (근거) New {int(new.customers):,}명은 관측 기간 주문이 1회이며 평균 구매액은 £{new.mean_monetary:,.2f}이다. "
        "(실행) 최근 1회 구매 고객에게 구매 7~14일 뒤 관련 상품 안내를 제공해 두 번째 주문 전환율 개선을 목표로 한다. "
        "(검증) 가입일과 관측 이전 구매 이력이 없으므로 진짜 신규 고객 여부는 알 수 없다. 추가 이력을 확인하고 캠페인 대조 실험으로 30일 두 번째 구매율을 비교한다.",
    ]


def write_readme(a, rfm, summary, insights):
    n = len(a.raw)
    table = summary.round(2).reset_index()
    text = f"""# 쇼핑몰 단골 찾기: 멀티모달 EDA와 RFM 분석

실제 거래 {n:,}행과 원본 8개 컬럼을 분석하는 재현 가능한 Python 프로젝트다. 전처리는 `DataAnalyzer` 클래스로 구성했고, 실행된 노트북에는 6종 시각화·수치 해석·고객 세분화를 포함했다.

## 데이터와 출처

- 거래: Chen, D. (2015), [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33). 라이선스 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 영국 온라인 소매업체의 2010-12-01~2011-12-09 자료이며 화폐는 GBP다.
- 원본 8열: `order_id`, `product_id`, `product_name`, `quantity`, `order_date`, `price`, `customer_id`, `country`. 원본 컬럼 순서와 의미를 유지하고 이름을 변경했다. 날짜는 Excel 일련번호에서 변환한다. country는 범주형이고 상품 카테고리를 뜻하지 않는다.
- **이미지는 실습용 합성 배열이다.** 원본에는 이미지가 없어 고정 seed 20261025로 상품별 16×16 RGB 배열을 생성했다. 실제 거래와 이미지가 연결된 공개 멀티모달 데이터셋은 아니다. 이미지 피처는 벡터 연산 검증용이며 시각 품질·매출의 인과관계 해석에 사용하지 않는다. 실제 상품 사진 요구가 엄격하다면 상품 ID에 연결된 정식 이미지를 추가해야 한다.
- `data/manifest.json`에 다운로드 주소·원본 ZIP 및 CSV SHA-256·합성 이미지 출처를 기록했다. 원본 데이터 전체를 사용하며 샘플링한 시계열이나 가짜 고객 데이터는 사용하지 않는다.

## 결과 요약

- 원본 {n:,}행 중 고객 ID 누락 {a.audit['missing_customer_rows']:,}행. 고객 ID는 추정하지 않는다.
- RFM 유효 구매 {a.audit['valid_purchase_rows']:,}행, 고객 {len(rfm):,}명, {rfm.segment.nunique()}개 세그먼트.
- 상품별 중앙값 및 전체 중앙값으로 가격 {a.audit['imputed_price']:,}행을 대치했다. 0 이하 가격은 EDA에서 결측으로 전환한다.
- 가격 IQR 이상치 {a.audit['price_iqr']['count']:,}행. 경계 £{a.audit['price_iqr']['lower']:.2f}~£{a.audit['price_iqr']['upper']:.2f}. 클리핑한 `price_clipped`는 시각화 전용이다.
- RFM 기준일: **{a.reference_date:%Y-%m-%d}**, 마지막 유효 거래의 달력 날짜. Recency는 일수, Frequency는 고유 주문 번호 수, Monetary는 원본 양수 가격×양수 수량의 합이다.
- 취소 주문(C 접두어), 수량 0 이하, 가격 0 이하/누락, 주문·상품·고객 ID 또는 날짜 누락을 RFM에서 제외한다. 원본 고객별 합계는 EDA 대치·클리핑으로 변경하지 않는다.
- Monetary와 월별 그래프는 **식별 고객의 총 구매액(gross purchases)**이며 반품 차감 순매출·이익·LTV가 아니다. 중복 의심 {a.audit['exact_duplicate_rows']:,}행은 합법적인 반복 상품 행일 수도 있어 표시만 하고 유지했다.

{markdown_table(table)}

## 세분화 규칙

R/F/M을 평균 동순위의 경험적 백분위로 1~5점화한다. R은 방향을 반전한다. 동점은 같은 점수를 받고, 모든 점수 구간이 동일 인원으로 채워질 필요는 없다. 다음 우선순위로 상호 배타적으로 배정한다.

1. VIP: R≥4, F≥4, M≥4
2. Churned: R≤2 (실제 이탈 확정이 아닌 상대적 미구매 장기화)
3. New: R≥4이고 관측 주문 1회
4. Loyal: 위 조건에 해당하지 않으면서 F≥3
5. Potential: 나머지

## 비즈니스 인사이트

""" + "\n\n".join(insights) + """

위 효과는 실험 가설이며 현재 관측 데이터로 검증된 성과가 아니다. 가격·수량·금액은 기계적 관계가 있으므로 상관계수를 인과관계로 해석하지 않는다. 2011년 12월은 9일까지의 부분 월이므로 전월 대비 하락을 수요 감소로 해석하지 않는다.

## 시각화

| 유형 | 결과 | 읽는 법 |
| --- | --- | --- |
| 히스토그램 | ![](outputs/figures/01_histogram.png) | log10 가격의 긴 꼬리 |
| 박스플롯 | ![](outputs/figures/02_boxplot.png) | IQR 처리 전후, 로그 가격 축 |
| 막대그래프 | ![](outputs/figures/03_bar.png) | RFM 그룹별 고객 수 |
| 히트맵 | ![](outputs/figures/04_heatmap.png) | Pearson 상관, 합성 이미지 포함 |
| 산점도 | ![](outputs/figures/05_scatter.png) | 고정 15,000행 표본, log10 수량·금액 |
| 라인차트 | ![](outputs/figures/06_line.png) | 고객 식별 가능한 총 구매액의 월별 추이 |

## 실행 방법

Python 3.10 이상 권장(검증: 3.12). 분석에 직접 사용하는 외부 라이브러리는 NumPy, Pandas, Matplotlib, Seaborn뿐이다. 이미지 처리는 전부 NumPy로 구현했다. Matplotlib이 간접 의존하는 Pillow는 프로젝트 코드에서 호출하지 않는다. XLSX 변환은 표준 라이브러리 zipfile/XML을 사용해 openpyxl이 필요 없다. Jupyter 패키지는 노트북 실행 도구다.

```powershell
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt
# CSV와 이미지 배열이 이미 포함되어 있어 다운로드 없이 분석 가능
.venv\\Scripts\\python.exe scripts/build_report.py --execute
.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

원본부터 다시 준비하려면 `.venv\\Scripts\\python.exe scripts/prepare_data.py`를 먼저 실행한다(원본 ZIP이 없으면 약 24MB를 다운로드). Linux/macOS는 가상환경 Python 경로를 `.venv/bin/python`으로 바꾼다. `--execute` 없이 실행하면 차트·CSV·README와 실행 전 노트북을 재생성한다. 실행 결과가 포함된 노트북은 `--execute`로 만든다. VS Code에서 `.venv` 커널로 `notebooks/analysis_report.ipynb`를 열어 직접 실행해도 된다.

## 파일 구조와 검증

- `src/pipeline.py`: 로드·그룹 대치·벡터 이미지/텍스트 피처·수동 IQR·RFM·요약 통계
- `notebooks/analysis_report.ipynb`: 한국어 해석, 코드, 출력과 6종 차트
- `scripts/prepare_data.py`: 원본 출처를 보존한 CSV 변환·합성 이미지 생성
- `scripts/build_report.py`: 차트·보고서·수치 근거의 일괄 재생성
- `outputs/`: 고객별 RFM, 세그먼트 요약, 기술 통계, 상관행렬, 품질 감사 JSON
- `tests/test_pipeline.py`: 주문 중복 집계·반품 제외·원본 금액 보존·결측 대치·이미지 연산·동점·경계값 검증

추가 이미지 피처로 NumPy 인접 픽셀 차이 기반 `image_edge`도 구현했다. 코호트·Plotly 보너스는 범위에 포함하지 않았다. 고객 연락처·가입일·마진 정보가 없어 실제 발송 또는 LTV 분석은 수행하지 않는다.

## GitHub 제출

제출 저장소: [00skgun/codessey_advance](https://github.com/00skgun/codessey_advance). 이 분석은 저장소의 `1-1/`에 있다. 실행 명령은 `1-1` 폴더를 기준으로 한다. `.venv`, 임시 파일, 다운로드 ZIP은 `.gitignore`로 제외되며 재현에 필요한 CSV·NPZ·실행된 노트북·차트는 포함한다.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def write_notebook(a, rfm, summary, insights, execute=False):
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("# 쇼핑몰 단골 찾기\n멀티모달 피처 엔지니어링 · EDA · RFM 고객 세분화\n\n"
       "실제 거래와 **별도 생성한 합성 이미지**를 결합한 실습이다. 이미지는 실제 상품 사진이 아니므로 이미지·매출의 관계를 비즈니스 근거로 사용하지 않는다. "
       "데이터: Chen (2015), [UCI Online Retail](https://doi.org/10.24432/C5BW33), CC BY 4.0. 원본 541,909행, 8열 전체를 사용한다.")
    code("from pathlib import Path\nimport sys\nimport io\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\n\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom src.pipeline import DataAnalyzer\nfrom scripts.build_report import create_figures\n\npd.set_option('display.max_columns', 20)\na = DataAnalyzer(ROOT / 'data/raw/transactions.csv')\na.load_data()\nprint('Rows / raw columns:', a.raw.shape)\ndisplay(a.raw.head())")
    md("## 1. 데이터 구조와 결측\n숫자(가격·수량), 텍스트(상품명), 범주(국가), 날짜를 확인한다. 원본 고객 ID 누락 "
       f"{a.audit['missing_customer_rows']:,}행은 RFM에서 제외한다. 따라서 분석 결과는 고객 식별이 가능한 거래를 대표하며 전체 방문 고객을 대표하지 않는다.")
    code("a.raw.info()\ndisplay(a.raw.describe(include='all'))\ndisplay(pd.DataFrame({'raw_missing': a.raw.isna().sum(), 'eda_missing_before': a.df.isna().sum()}))")
    md("## 2. 그룹별 결측치 대치\n0 이하 가격을 EDA에서 결측으로 바꾸고 **동일 상품 ID별 중앙값 → 전체 중앙값** 순서로 대치한다. "
       f"대치 대상은 {a.audit['imputed_price']:,}행이다. 상품명은 동일 상품의 관측값을 앞뒤로 채우고 남으면 UNKNOWN PRODUCT를 사용한다. "
       "고객 ID와 거래일은 채우지 않으며, RFM은 대치 이전 원본을 사용한다.")
    code("a.handle_missing_values(strategy='median', group_col='product_id')\nprint('Imputed price rows:', a.audit['imputed_price'])\ndisplay(a.df.isna().sum().rename('remaining_missing').to_frame())")
    md("## 3. NumPy 멀티모달 피처\n(N,H,W,C) 배열의 축 (1,2,3)을 한 번에 축약해 이미지별 평균·표준편차를 계산한다. "
       "전체 이미지를 벡터로 처리하므로 이미지/픽셀 for 문이 없다. 표준편차는 모집단 기준 ddof=0이고, 텍스트는 길이와 공백 기준 단어 수로 수치화한다. "
       "추가로 float32 변환 후 인접 픽셀 절댓값 차이로 엣지를 계산해 uint8 차감의 오버플로를 피한다.")
    code("with np.load(ROOT / 'data/image_fixtures.npz', allow_pickle=False) as fixture:\n    print('Synthetic image batch:', fixture['images'].shape)\n    a.engineer_features(fixture['product_ids'], fixture['images'])\ndisplay(a.df[['product_name', 'name_length', 'name_word_count', 'image_mean', 'image_std', 'image_edge']].head())")
    md("## 4. 직접 구현한 IQR 이상치 탐지\nQ1−1.5×IQR 미만 또는 Q3+1.5×IQR 초과를 표시한다. "
       f"가격 Q1=£{a.audit['price_iqr']['q1']:.2f}, Q3=£{a.audit['price_iqr']['q3']:.2f}, "
       f"상한=£{a.audit['price_iqr']['upper']:.2f}이며 이상치는 {a.audit['price_iqr']['count']:,}행이다. "
       "고가 상품은 유효한 구매일 수 있으므로 자동 삭제하지 않는다. 전후 박스플롯은 파생 열 클리핑의 영향만 보여준다.")
    code("outliers = a.detect_outliers('price', threshold=1.5, treatment='clip')\nprint(a.audit['price_iqr'])\ndisplay(outliers[['product_id', 'price', 'price_clipped']].head())")
    stats = a.summary_statistics()
    corr = a.df[["price", "quantity", "amount", "name_length", "image_mean", "image_std"]].corr()
    md("## 5. 기술 통계와 상관분석\n아래 통계는 대치된 EDA 전체 상품 행 기준이며 반품 수량도 포함한다. "
       f"가격 평균 £{stats.loc['price','mean']:.2f}, 중앙값 £{stats.loc['price','50%']:.2f}, "
       f"표준편차 £{stats.loc['price','std']:.2f}, Q1 £{stats.loc['price','25%']:.2f}, Q3 £{stats.loc['price','75%']:.2f}이다. "
       "평균과 중앙값의 차이는 고가 꼬리의 영향을 보여주므로 대표 가격은 중앙값과 사분위수도 함께 본다.\n\n"
       f"① 수량–금액 Pearson r={corr.loc['quantity','amount']:.4f}. 금액=가격×수량이라는 정의와 대량 주문의 영향이 있으므로 수량과 금액을 독립적인 수요 지표처럼 해석하면 안 된다.\n\n"
       f"② 가격–수량 Pearson r={corr.loc['price','quantity']:.4f}. 이 선형 상관만으로 가격 인하가 수요를 늘린다고 결론낼 수 없으므로 상품별·기간별 통제 분석이 필요하다. "
       "합성 이미지의 상관은 생성 과정에 따른 우연이며 상품 사진의 효과에 대한 근거가 아니다.")
    code("display(a.summary_statistics())\ncorrelation_columns = ['price', 'quantity', 'amount', 'name_length', 'image_mean', 'image_std']\ndisplay(a.df[correlation_columns].corr())")
    md("## 6. RFM과 회계 기준\n빈도는 상품 행 수가 아닌 `order_id.nunique()`이고 금액은 원본 양수 거래의 합이다. "
       f"기준일 {a.reference_date:%Y-%m-%d}, 유효 구매 {a.audit['valid_purchase_rows']:,}행, 고객 {len(rfm):,}명이다. "
       "취소·비양수 가격/수량·식별자/날짜 결측은 제외한다. 반품 차감 순매출이 아니므로 수익성 판단에는 추가 데이터가 필요하다.\n\n"
       "동점은 평균 백분위로 같은 점수를 부여한다. 우선순위는 VIP(R/F/M≥4), Churned(R≤2), New(R≥4·주문 1회), Loyal(F≥3), Potential(나머지)이다.")
    code("rfm = a.calculate_rfm()\nsegment_summary = rfm.groupby('segment').agg(customers=('monetary', 'size'), mean_recency=('recency', 'mean'), mean_frequency=('frequency', 'mean'), mean_monetary=('monetary', 'mean'), gross_purchases=('monetary', 'sum'))\ndisplay(rfm.head())\ndisplay(segment_summary.round(2))\nprint('Reference date:', a.reference_date)\nprint('RFM customers:', len(rfm), 'Segments:', rfm.segment.nunique())")
    md("## 7. 필수 6종 시각화\n히스토그램·박스플롯의 가격은 양수 대치값을 사용한다. 산점도와 월별 라인차트는 원본 유효 구매만 사용한다. "
       "로그 축과 log10 변환은 축 레이블에 명시했다. 이미지 피처가 합성이라는 점도 히트맵에 표시했다.")
    code("%matplotlib inline")
    code("figures = create_figures(a, rfm)\nfor name, fig in figures.items():\n    display(fig)\n    plt.close(fig)")
    purchase = a.valid_transactions()
    monthly = purchase.groupby(purchase.order_date.dt.to_period("M")).amount.sum()
    md(f"### 시계열 해석\n2011년 11월 식별 고객 총 구매액은 £{monthly.loc['2011-11']:,.2f}, "
       f"2011년 12월은 £{monthly.loc['2011-12']:,.2f}이다. 12월은 1~9일만 관측했으므로 월 합계 감소를 매출 부진으로 해석할 수 없다. "
       "동일 일수 또는 전년 동기 비교가 필요하다. 이 시계열은 2010~2011년 이력으로 현재 시장을 설명하지 않는다.")
    md("## 8. 세그먼트별 실행 전략\n\n" + "\n\n".join(insights))
    md("## 9. 결과 검증과 한계\n세그먼트 수·고객 유일성·일수 비음수·원본 구매액 보존을 검증한다. "
       "결측 제외 사유는 중복될 수 있으므로 항목별 제외 건수를 더해서 전체 제외 행 수로 사용하지 않는다. "
       "가입 시점, 실제 상품 이미지, 마진, 노출/전환 이력, 고객 연락처는 제공되지 않았다.")
    code("purchase = a.valid_transactions()\nassert len(rfm) == purchase.customer_id.nunique()\nassert rfm.index.is_unique and rfm.recency.ge(0).all()\nassert rfm.segment.nunique() >= 4\nassert np.isclose(rfm.monetary.sum(), purchase.amount.sum())\nassert a.df.shape[0] >= 1000 and a.raw.shape[1] >= 8\nassert a.df[['price', 'image_mean', 'image_std', 'name_length']].notna().all().all()\nprint('All dataset acceptance checks passed.')\ndisplay(pd.Series(a.audit, name='value').to_frame())")
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (project .venv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    path = ROOT / "notebooks/analysis_report.ipynb"
    nbf.write(nb, path)
    if execute:
        # A project-local kernelspec avoids using a broken global Python kernel.
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        kernel_root = ROOT / "tmp/kernels"
        kernel = kernel_root / "python3"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Project Python", "language": "python"}), encoding="utf-8")
        ksm = KernelSpecManager(kernel_dirs=[str(kernel_root)])
        manager = KernelManager(kernel_name="python3", kernel_spec_manager=ksm)
        client = NotebookClient(nb, timeout=600, kernel_name="python3", km=manager,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbf.write(nb, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    (ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "notebooks").mkdir(exist_ok=True)
    a, rfm, summary, missing = analyze()
    rfm.to_csv(ROOT / "outputs/customer_rfm.csv")
    summary.to_csv(ROOT / "outputs/segment_summary.csv")
    a.summary_statistics().to_csv(ROOT / "outputs/descriptive_statistics.csv")
    a.df[["price", "quantity", "amount", "name_length", "image_mean", "image_std"]].corr().to_csv(ROOT / "outputs/correlation.csv")
    pd.concat([missing.rename("before"), a.df.isna().sum().rename("after")], axis=1).to_csv(ROOT / "outputs/missing_values.csv")
    figures = create_figures(a, rfm)
    for name, fig in figures.items():
        fig.savefig(ROOT / "outputs/figures" / (name + ".png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    (ROOT / "outputs/quality_audit.json").write_text(json.dumps(a.audit, indent=2), encoding="utf-8")
    insights = interpretations(a, rfm, summary)
    write_readme(a, rfm, summary, insights)
    write_notebook(a, rfm, summary, insights, execute=args.execute)
    print(summary.round(2).to_string())
    print("Notebook executed:", args.execute)


if __name__ == "__main__":
    main()
