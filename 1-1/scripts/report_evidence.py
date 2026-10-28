"""Shared calculations and Korean explanations for README and notebook evidence."""
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    default = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if path is not None:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(overrides) - set(default)
        if unknown:
            raise ValueError("Unknown configuration keys: " + str(sorted(unknown)))
        default.update(overrides)
    if default["missing_strategy"] not in {"median", "mean"}:
        raise ValueError("missing_strategy must be median or mean")
    threshold = default["iqr_threshold"]
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("iqr_threshold must be finite and nonnegative")
    for key in ["scatter_sample_size", "benchmark_image_count", "benchmark_repeats"]:
        if type(default[key]) is not int or default[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if type(default["random_seed"]) is not int or default["random_seed"] < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    return default


def correlation_pairs(analyzer):
    """Explicit pairwise Pearson calculations, distinct from heatmap rendering."""
    rows = []
    for x, y in [("quantity", "amount"), ("price", "quantity"), ("price", "amount")]:
        pair = analyzer.df[[x, y]].dropna()
        rows.append({"x": x, "y": y, "n": len(pair), "pearson_r": pair[x].corr(pair[y])})
    return pd.DataFrame(rows)


def save_figure_evidence(figures):
    kinds = ["histogram", "boxplot", "bar", "heatmap", "scatter", "line"]
    records = []
    for (name, fig), kind in zip(figures.items(), kinds):
        ax = fig.axes[0]
        labels = {"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel()}
        if not all(labels.values()):
            raise ValueError("Figure title and axis labels are required: " + name)
        path = ROOT / "outputs/figures" / (name + ".png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        records.append({"type": kind, "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **labels})
    if len(records) != 6:
        raise ValueError("Exactly six required figure types must be saved")
    (ROOT / "outputs/figure_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def supplemental_sections(a, benchmark):
    pairs = correlation_pairs(a)
    corr = pairs.set_index(["x", "y"]).pearson_r
    imp = a.audit["price_imputation"]
    details = a.imputation_details["price"]
    worst = details.sort_values(["missing_ratio", "missing_rows"], ascending=False).iloc[0]
    worst_id = details.sort_values(["missing_ratio", "missing_rows"], ascending=False).index[0]
    original = a.df.price
    clipped = a.df.price_clipped
    q = a.audit["price_iqr"]
    normal = NormalDist()
    theoretical_fence = normal.inv_cdf(.75) * (1 + 2 * a.analysis_config["iqr_threshold"])
    normal_tail_pct = (1 - normal.cdf(theoretical_fence)) * 200
    dtype_counts = a.raw.dtypes.astype(str).value_counts()
    types = ", ".join(f"`{name}` {count}열" for name, count in dtype_counts.items())
    raw_image_bytes = 4070 * 16 * 16 * 3
    slowest_stage, slowest_seconds = max(a.audit["stage_seconds"].items(), key=lambda item: item[1])
    return f"""## 실행 증거와 평가 항목 안내

- [실행된 노트북](notebooks/analysis_report.ipynb): 모든 코드 셀의 실행 번호와 6개 PNG 출력을 저장한다.
- [전체 실행 로그](outputs/execution.log), [실행 상태·버전·SHA-256](outputs/execution_summary.json): 성공 여부와 실제 실행 셀 수·오류 수를 확인한다.
- [차트 6종](outputs/figures), [차트 제목·축 레이블·파일 체크섬](outputs/figure_manifest.json): 파일이 실제 존재하는지 추적할 수 있다.
- [19개 평가 항목 대응표](outputs/evaluation_evidence.md): 요구 사항과 검토할 파일을 한곳에 모았다.
- 원본 정규화 후 타입 분포: {types}. 원본 8열에서 날짜·숫자·문자/범주를 다루고 별도 배열 이미지 피처를 결합한다.

## 핵심 메서드와 설정

| 메서드 | 책임 | 원본 회계 데이터 변경 |
| --- | --- | --- |
| `load_data()` | 8열 검증·숫자/날짜 변환·원본/EDA 분리 | 로드 시 원본 사본 생성 |
| `handle_missing()` | 평가 기준 호환 진입점; 아래 메서드에 위임 | 없음 |
| `handle_missing_values()` | 그룹 통계 대치·전체 폴백·그룹별 결측 감사 | 없음 |
| `image_features()` / `engineer_features()` | 벡터 이미지 통계·텍스트 길이/단어 수 | 없음 |
| `detect_outliers()` | IQR 플래그·별도 클리핑 열 | 없음 |
| `calculate_rfm()` | 식별 가능한 원본 양수 구매의 고객별 RFM | 없음 |

파라미터는 [config.json](config.json)에 모았다. `--config path/to/override.json`으로 대치 전략/그룹, IQR 배수, RFM 기준일, 산점도 표본·seed와 벤치마크 크기를 변경할 수 있다. 메서드 인자로 직접 전달하는 방식도 유지한다. `outputs/config_used.json`은 이번 실행의 실제 설정이다.

## 상관계수 2쌍 이상과 기술 통계 해석

분석 범위는 **대치 후 전체 상품 행(반품 포함)**이며 클리핑 전 `price`, `quantity`, `amount`다. [상관계수 직접 계산 결과](outputs/correlation_pairs.csv)에서 각 쌍의 표본 수와 값을 확인할 수 있다.

| 변수 쌍 | Pearson r | 해석과 시사점 |
| --- | --- | --- |
| quantity / amount | {corr.loc[('quantity', 'amount')]:.6f} | 금액=가격×수량의 기계적 관계와 대량 주문이 영향을 준다. 수량과 금액을 서로 독립적인 수요 근거로 중복 계산하지 않는다. |
| price / quantity | {corr.loc[('price', 'quantity')]:.6f} | 전체 행에서 선형 관계가 약해도 개별 상품의 가격 탄력성이 없다는 뜻은 아니다. 상품·기간·반품 여부를 통제한 분석이 필요하다. |
| price / amount | {corr.loc[('price', 'amount')]:.6f} | 음수 반품과 고가 행의 영향이 섞인다. 거래 부호를 나눈 민감도 분석 없이 가격의 매출 효과로 해석하지 않는다. |

가격 평균 £{original.mean():.4f}, 중앙값 £{original.median():.4f}, 표준편차 £{original.std():.4f}, Q1 £{q['q1']:.2f}, Q3 £{q['q3']:.2f}다. 오른쪽 꼬리에 민감한 평균·표준편차와 함께 중앙값·사분위수를 제시해야 대표 가격을 과장하지 않는다. [전체 기술 통계](outputs/descriptive_statistics.csv)에는 수량·금액·텍스트·이미지 통계도 포함한다.

## 그룹별 결측 대치의 근거와 위험

대상 가격 결측/비양수는 {a.audit['imputed_price']:,}행({a.audit['imputed_price']/len(a.df)*100:.3f}%)이다. 총 {imp['groups']:,}개 상품 그룹 중 {imp['groups_with_missing']:,}개에서 대치가 필요했다. 그룹 내 관측 가격으로 {imp['group_imputed_rows']:,}행을 채우고, 가격 전체가 결측인 {imp['all_missing_groups']:,}개 그룹의 {imp['global_fallback_rows']:,}행은 전체 {imp['strategy']} £{imp['global_reference_value']:.4f}로 대치했다. 상품 `{worst_id}`의 결측률은 {worst.missing_ratio*100:.1f}%({int(worst.missing_rows):,}/{int(worst.total_rows):,}행)다. [상품별 결측률·관측 수·대치값](outputs/imputation_by_product.csv)에 전체 그룹을 기록한다.

상품마다 가격대가 달라 동일 상품 통계가 전체 통계보다 자연스러운 기준이다. 기본 중앙값은 비싼 꼬리값의 영향을 줄이지만 **결측이 무작위라는 사실을 보장하지 않는다.** 예를 들어 가격 10·20에서 결측을 15로 채우면 모집단 분산이 25에서 16.67로 줄어든다(설명용 예시). 그룹 내 분산이 줄어 그룹 간 차이가 상대적으로 과장되고, 할인 기간에만 가격이 누락되었다면 해당 그룹 가격을 과대 추정할 수 있다. 전체 폴백은 희소·신규 상품의 특성을 지울 수도 있다.

실제 관측 가격만의 모집단 분산은 {imp['observed_variance_ddof0']:.4f}, 대치 후 전체 분산은 {imp['filled_variance_ddof0']:.4f}다. 두 모집단의 구성·크기가 다르므로 이 차이를 대치의 인과 효과로 해석하지 않는다. 관측/대치 여부 플래그를 보존하고 상품별 결측률·유효 표본 수에 따라 민감도 분석하며, 학습으로 확장할 때는 **훈련 구간에서만** 대치 통계를 추정한다. RFM에는 대치 금액을 사용하지 않는다.

피처 결측 정책: 상품명은 같은 상품의 관측 문자열→UNKNOWN PRODUCT, 이미지 누락·NaN/무한대·[0,255] 밖 값은 오류로 중단한다. 이미지 극단값 0/255는 유효한 픽셀로 유지한다. 텍스트 길이·이미지 피처를 임의 클리핑하지 않고, 가격 극단값은 별도 플래그와 시각화용 열로 구분한다.

## IQR의 강건성과 비대칭 분포 한계

평균±표준편차 기준은 극단값 때문에 평균과 경계가 함께 움직일 수 있다. IQR은 중앙 50%의 사분위수를 써 이에 덜 민감하고 정규성 가정 없이 탐색할 수 있다. 다만 정규분포에서도 이상치 확정 검정이 아니다. 표준정규분포에 현재 {a.analysis_config['iqr_threshold']:g}×IQR 규칙을 적용하면 이론 경계는 약 ±{theoretical_fence:.3f}σ이고 정상 관측값 약 {normal_tail_pct:.3f}%가 경계 밖에 놓인다(표준정규 CDF로 계산한 양측 이론값).

현재 가격의 경계는 £{q['lower']:.2f}~£{q['upper']:.2f}이고 {q['count']:,}행({q['count']/len(a.df)*100:.2f}%)이 표시된다. 오른쪽으로 치우친 양수 가격에서 음수 하한은 실용성이 낮고 상한은 정상 고가 상품까지 과도하게 표시할 수 있다. 따라서 오류라고 단정하거나 삭제하지 않는다. 상품군별 IQR·log 가격·강건 통계 및 실제 업무 기준을 함께 검토하고, IQR=0인 그룹에서는 사분위수 밖 값이 모두 표시될 수 있음도 확인한다. [NIST의 IQR 경계 설명](https://itl.nist.gov/div898/handbook/eda/section3/boxplot.htm)과 [비대칭 분포의 과다 탐지 한계](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/boxplot.htm)를 참고했다.

클리핑 전후 평균은 £{original.mean():.4f}→£{clipped.mean():.4f}, 표준편차는 £{original.std():.4f}→£{clipped.std():.4f}다. 분산 감소는 클리핑의 기계적 결과이며 분석 정확도 개선을 증명하지 않는다. 원본 가격·RFM Monetary는 보존한다.

## NumPy 속도 원리·벤치마크·대규모 병목

`np.ascontiguousarray(..., dtype=np.float32)`로 C 연속 메모리와 일관된 strides를 확보하고 축 (1,2,3)을 한 번에 축약한다. Python이 이미지/픽셀마다 함수를 호출하는 비용을 줄이고 컴파일된 내부 루프가 연속 메모리에 접근해 캐시 지역성을 높일 수 있다. NumPy는 CPU·빌드·연산 종류에 따라 SIMD 커널을 런타임 선택한다. 모든 mean/std가 SIMD로 실행되거나 자동 멀티코어가 된다고 보장하지 않는다. [NumPy 1.26 CPU/SIMD 문서](https://numpy.org/doc/1.26/reference/simd/index.html).

이번 환경의 동일 float32 평균/표준편차 연산({benchmark['images']:,}개 이미지, {benchmark['repeats']}회 중앙값)은 벡터 {benchmark['vectorized_median_ms']:.3f}ms, 이미지별 Python 루프 {benchmark['loop_median_ms']:.3f}ms, 비율 {benchmark['speedup']:.2f}배다. 최대 출력 차이는 {benchmark['max_absolute_difference']:.8f}다. [실측 조건·각 반복 시간](outputs/vectorization_benchmark.json)을 저장했다. 이는 I/O·형변환·엣지·DataFrame 결합을 제외한 특정 환경 측정값으로 성능 보장이 아니다. 비교용 루프는 `scripts/benchmark_vectorization.py`에만 있으며 실제 `image_features()`에는 이미지/픽셀 반복문이 없다.

현재 4,070×16×16×3 uint8 배열 자체는 약 {raw_image_bytes/1024**2:.2f}MiB이고 float32 변환본은 약 {raw_image_bytes*4/1024**2:.2f}MiB다(임시 배열·DataFrame 제외 추정). 실제 단계별 시간은 [quality_audit.json](outputs/quality_audit.json)의 `stage_seconds`에 기록했다. 이번 측정에서 가장 오래 걸린 분석 단계는 `{slowest_stage}`({slowest_seconds:.3f}초)다. 단계별 경과 시간만으로 CPU와 I/O를 분리해 확정할 수는 없다. 규모가 커지면 CSV 문자열 파싱·원본 사본·GroupBy/Join과 이미지 압축 해제·float32 사본·std/np.diff 임시 배열이 각각 I/O·메모리 대역폭·RAM 병목 후보가 된다. 현재 크기에서는 무조건적인 병렬화보다 실측이 우선이다.

개선안: 상품당 피처를 한 번만 계산하고 ID로 결합한다. 224×224 RGB 이미지는 64개 배치의 float32 입력만 약 36.75MiB이므로 임시 배열까지 고려해 배치 크기를 제한한다. 대량 이미지 파일 읽기는 작업자 수를 제한한 I/O 스레드와 배치 프리패치를 검토하고, CPU 병렬화는 작업자별 메모리 복제·스레드 과다 경쟁을 측정해 적용한다. 압축 NPZ 대신 비압축 NPY+`mmap_mode='r'`로 필요한 배치를 읽을 수 있다. 시각화는 기간·국가별 층화 표본을 검토하되 RFM 합계와 고객 수에는 전체 거래를 사용한다. 현재 실행은 전체 이미지 벡터 연산이며 배치/병렬 확장은 제안 단계다.

## 세그먼트 실험 KPI와 점수 컷오프

1~5점은 서로 다른 R/F/M 단위를 비교하기 쉬운 초기 5분위 운영 규칙이다. VIP≥4는 각 지표의 상대적 상위 약 40%를 교집합으로 좁히려는 출발점이며, 동점 때문에 정확히 40%는 아니다. 최적 이탈 임계값으로 검증된 것이 아니므로 시즌성·마진·캠페인 수용 능력과 실험 결과로 재조정한다.

| 대상 | 제안 실험·관측 기간 | 1차 지표 | 제안 최소 개선 목표 |
| --- | --- | --- | --- |
| VIP (우선) | 무작위 혜택/대조군, 60일 | 1회 이상 재구매한 고객 수 / 배정 고객 수 | 대조군 대비 +3%p |
| Churned | 재입고 안내/대조군, 30일 | 복귀 고객 수 / 배정 고객 수 | +2%p |
| New | 관련 상품 안내/대조군, 30일 | 두 번째 주문 고객 수 / 배정 고객 수 | +5%p |
| Loyal | 상품 추천/대조군, 30일 | 재구매 고객 수 / 배정 고객 수 | +2%p |
| Potential | 소규모 추천 실험, 30일 | 재구매 고객 수 / 배정 고객 수 | +2%p |

위 숫자는 **제안 목표**이며 현재 관측된 캠페인 효과가 아니다. 실제 기준 전환율·표본 수·검정력으로 실험 가능성을 먼저 확인하고, 공헌이익/고객이 악화되지 않는지도 함께 본다. 노출이 없어도 원래 배정한 군을 분모로 평가하며 캠페인 전후 단순 비교로 인과효과를 주장하지 않는다.

## ML 확장 예시: 향후 90일 미구매 예측

명시적 타겟은 기준일 t에 기존 구매 이력이 있는 고객의 **(t, t+90일] 내 유효 구매가 없으면 `no_purchase_90d=1`, 있으면 0**이다. 이는 관측상 미구매이지 영구 이탈 확정이 아니다. 모델을 학습하거나 예측 정확도를 산출한 프로젝트는 아니며 향후 설계 예시다.

권장 입력은 t 이전 180일 R/F/M, 평균 주문액, 최초 관측 구매 이후 일수, 고유 상품 수, 고객별 구매 상품명 길이/단어 수의 평균, 국가 등이다. 실제 상품 사진을 확보한 후에만 구매 이력으로 집계한 이미지 평균·표준편차·엣지를 후보로 삼는다. **현재 합성 이미지 피처와 원시 customer_id는 입력에서 제외**한다. 행별 `amount`는 과거 합계/평균으로 고객 단위에 맞춰 집계한다.

마지막 관측일이 2011-12-09이므로 t+90일이 이 날짜 이내인 고객/기준일만 정답을 만들 수 있다. 최신 RFM 스냅샷에 미래 정답을 억지로 붙이거나 관측 종료까지 미구매한 고객 전부를 1로 두면 안 된다. 실제 후속 구매로 정답을 만들고 시간순 훈련/검증 분리, 훈련 데이터만의 대치·스케일 추정, 고객/기간 중복으로 인한 누출을 통제한다. 현재 RFM 세그먼트를 정답으로 두고 동일 RFM을 입력하면 규칙 복제에 그치므로 별도 미래 타겟을 쓴다.
"""


def write_evaluation_index():
    rows = [
        (1, "handle_missing 호환 진입점", "../src/pipeline.py"),
        (2, "무에러 노트북 전체 실행 및 셀/PNG 수", "execution_summary.json"),
        (3, "행·열·dtype 수치와 info/head/describe", "data_structure.txt"),
        (4, "NumPy 이미지/텍스트 파생·결측/극단값 정책", "../README.md"),
        (5, "IQR 전후 실제 박스플롯", "figures/02_boxplot.png"),
        (6, "6종 이미지와 제목·축 레이블", "figure_manifest.json"),
        (7, "직접 계산한 3쌍 상관과 수치 해석", "correlation_pairs.csv"),
        (8, "5개 고객 세그먼트와 KPI", "segment_summary.csv"),
        (9, "클래스 메서드별 책임 표", "../README.md"),
        (10, "벡터/루프 동일 연산 실측 비교", "vectorization_benchmark.json"),
        (11, "분석 파라미터 중앙 관리", "../config.json"),
        (12, "상품별 결측률·관측 수·글로벌 폴백", "imputation_by_product.csv"),
        (13, "정규분포 이론 비교·비대칭 IQR 한계", "../README.md"),
        (14, "대치 분산 축소·그룹 편향 예시", "../README.md"),
        (15, "RFM 컷오프 선택과 검증 한계", "../README.md"),
        (16, "연속 메모리·캐시·CPU/SIMD 설명", "../README.md"),
        (17, "실측 단계 시간·메모리 추정·배치/I/O 개선", "quality_audit.json"),
        (18, "VIP 등 세그먼트별 실험 목표·추가 데이터", "../README.md"),
        (19, "향후 90일 미구매 타겟·입력 피처·누출 통제", "../README.md"),
    ]
    lines = ["# 평가 피드백 19개 항목 대응", "", "평가 결과를 대신 판정하는 표가 아니라 구현·해석·실행 증거의 위치를 안내한다.", "",
             "| 항목 | 반영 내용 | 증거 |", "| --- | --- | --- |"]
    lines += [f"| {number} | {description} | [확인]({path}) |" for number, description, path in rows]
    (ROOT / "outputs/evaluation_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
