# 쇼핑몰 단골 찾기: 멀티모달 EDA와 RFM 분석

실제 거래 541,909행과 원본 8개 컬럼을 분석하는 재현 가능한 Python 프로젝트다. 전처리는 `DataAnalyzer` 클래스로 구성했고, 실행된 노트북에는 6종 시각화·수치 해석·고객 세분화를 포함했다.

## 데이터와 출처

- 거래: Chen, D. (2015), [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33). 라이선스 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 영국 온라인 소매업체의 2010-12-01~2011-12-09 자료이며 화폐는 GBP다.
- 원본 8열: `order_id`, `product_id`, `product_name`, `quantity`, `order_date`, `price`, `customer_id`, `country`. 원본 컬럼 순서와 의미를 유지하고 이름을 변경했다. 날짜는 Excel 일련번호에서 변환한다. country는 범주형이고 상품 카테고리를 뜻하지 않는다.
- **이미지는 실습용 합성 배열이다.** 원본에는 이미지가 없어 고정 seed 20261025로 상품별 16×16 RGB 배열을 생성했다. 실제 거래와 이미지가 연결된 공개 멀티모달 데이터셋은 아니다. 이미지 피처는 벡터 연산 검증용이며 시각 품질·매출의 인과관계 해석에 사용하지 않는다. 실제 상품 사진 요구가 엄격하다면 상품 ID에 연결된 정식 이미지를 추가해야 한다.
- `data/manifest.json`에 다운로드 주소·원본 ZIP 및 CSV SHA-256·합성 이미지 출처를 기록했다. 원본 데이터 전체를 사용하며 샘플링한 시계열이나 가짜 고객 데이터는 사용하지 않는다.

## 결과 요약

- 원본 541,909행 중 고객 ID 누락 135,080행. 고객 ID는 추정하지 않는다.
- RFM 유효 구매 397,884행, 고객 4,338명, 5개 세그먼트.
- 상품별 중앙값 및 전체 중앙값으로 가격 2,517행을 대치했다. 0 이하 가격은 EDA에서 결측으로 전환한다.
- 가격 IQR 이상치 39,854행. 경계 £-3.07~£8.45. 클리핑한 `price_clipped`는 시각화 전용이다.
- RFM 기준일: **2011-12-09**, 마지막 유효 거래의 달력 날짜. Recency는 일수, Frequency는 고유 주문 번호 수, Monetary는 원본 양수 가격×양수 수량의 합이다.
- 취소 주문(C 접두어), 수량 0 이하, 가격 0 이하/누락, 주문·상품·고객 ID 또는 날짜 누락을 RFM에서 제외한다. 원본 고객별 합계는 EDA 대치·클리핑으로 변경하지 않는다.
- Monetary와 월별 그래프는 **식별 고객의 총 구매액(gross purchases)**이며 반품 차감 순매출·이익·LTV가 아니다. 중복 의심 5,268행은 합법적인 반복 상품 행일 수도 있어 표시만 하고 유지했다.

| segment | customers | mean_recency | mean_frequency | mean_monetary | gross_purchases | customer_share_pct | spend_share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Churned | 1727 | 191.6 | 1.98 | 777.43 | 1342624.23 | 39.81 | 15.07 |
| Loyal | 1169 | 32.36 | 3.52 | 1441.2 | 1684766.78 | 26.95 | 18.91 |
| New | 236 | 18.42 | 1.0 | 325.84 | 76897.84 | 5.44 | 0.86 |
| Potential | 295 | 52.46 | 1.0 | 402.42 | 118712.85 | 6.8 | 1.33 |
| VIP | 911 | 11.86 | 11.5 | 6244.13 | 5688406.2 | 21.0 | 63.83 |

## 세분화 규칙

R/F/M을 평균 동순위의 경험적 백분위로 1~5점화한다. R은 방향을 반전한다. 동점은 같은 점수를 받고, 모든 점수 구간이 동일 인원으로 채워질 필요는 없다. 다음 우선순위로 상호 배타적으로 배정한다.

1. VIP: R≥4, F≥4, M≥4
2. Churned: R≤2 (실제 이탈 확정이 아닌 상대적 미구매 장기화)
3. New: R≥4이고 관측 주문 1회
4. Loyal: 위 조건에 해당하지 않으면서 F≥3
5. Potential: 나머지

## 비즈니스 인사이트

**VIP 유지** — (근거) VIP 911명은 고객의 21.0%이지만 식별 고객 총 구매액의 63.8%를 차지한다. (실행) VIP 고객을 대상으로 우선 재입고 알림과 전담 응대 혜택을 제공하여 재구매 유지율 개선을 목표로 한다. (검증) 실제 순이익·반품·캠페인 노출 데이터가 필요하며, 무작위 대조군 대비 60일 재구매율과 공헌이익이 개선되지 않으면 전략을 기각한다.

**휴면 위험 고객 복귀** — (근거) Churned 1,727명(39.8%)의 평균 미구매 기간은 191.6일이다. (실행) Churned 중 과거 구매액이 높은 고객에게 개인화 재입고 안내를 보내고, 소규모 쿠폰 실험으로 30일 복귀율 개선을 목표로 한다. (검증) Churned는 상대적 최근성 점수에 붙인 이름이며 실제 이탈 확정이 아니다. 수신 동의·시즌성·할인 비용 데이터를 확보하고 대조군 대비 증분 이익으로 검증한다.

**첫 관측 구매 고객의 두 번째 주문** — (근거) New 236명은 관측 기간 주문이 1회이며 평균 구매액은 £325.84이다. (실행) 최근 1회 구매 고객에게 구매 7~14일 뒤 관련 상품 안내를 제공해 두 번째 주문 전환율 개선을 목표로 한다. (검증) 가입일과 관측 이전 구매 이력이 없으므로 진짜 신규 고객 여부는 알 수 없다. 추가 이력을 확인하고 캠페인 대조 실험으로 30일 두 번째 구매율을 비교한다.

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
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# CSV와 이미지 배열이 이미 포함되어 있어 다운로드 없이 분석 가능
.venv\Scripts\python.exe scripts/build_report.py --execute
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

원본부터 다시 준비하려면 `.venv\Scripts\python.exe scripts/prepare_data.py`를 먼저 실행한다(원본 ZIP이 없으면 약 24MB를 다운로드). Linux/macOS는 가상환경 Python 경로를 `.venv/bin/python`으로 바꾼다. `--execute` 없이 실행하면 차트·CSV·README와 실행 전 노트북을 재생성한다. 실행 결과가 포함된 노트북은 `--execute`로 만든다. VS Code에서 `.venv` 커널로 `notebooks/analysis_report.ipynb`를 열어 직접 실행해도 된다.

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
