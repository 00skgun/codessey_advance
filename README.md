# codessey_advance

AI 심화 학습 과제를 모아 두는 저장소다. 상위 폴더 `codessey_advanced`를 저장소 루트로 사용하고, 개별 과제는 하위 폴더에 둔다.

## 1-1. 쇼핑몰 단골 찾기

실제 이커머스 거래 541,909건으로 전처리, 멀티모달 피처 엔지니어링, EDA, RFM 고객 세분화를 수행한다. 실습용 이미지 배열은 합성 데이터임을 분석 문서에 명시했다.

- [프로젝트 설명·분석 결과·데이터 출처](1-1/README.md)
- [실행된 분석 노트북](1-1/notebooks/analysis_report.ipynb)
- [DataAnalyzer 파이프라인](1-1/src/pipeline.py)
- [세그먼트별 분석 결과](1-1/outputs/segment_summary.csv)
- [피드백 19개 항목별 구현·실행 증거](1-1/outputs/evaluation_evidence.md)
- [노트북 전체 실행 로그](1-1/outputs/execution.log) · [실행 상태·셀 수·오류 수](1-1/outputs/execution_summary.json)
- [차트 6종 제목·축 레이블·체크섬](1-1/outputs/figure_manifest.json)

`handle_missing()` 호환 메서드, 상관계수 3쌍의 수치 해석, IQR·그룹 대치의 한계, NumPy 성능 실측, 세그먼트별 검증 KPI와 향후 90일 미구매 예측 설계를 과제 README·노트북에 함께 기록했다.

## 실행 방법

저장소 루트에서 과제 폴더로 이동한 후 실행한다. Python 3.12에서 검증했다.

```powershell
cd 1-1
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe scripts/build_report.py --execute
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

CSV와 이미지 배열을 포함하므로 분석 재실행 시 원본 데이터를 다운로드할 필요가 없다. `.venv`, 임시 파일, 원본 다운로드 ZIP, 생성된 제출용 ZIP은 Git 추적에서 제외한다.

## 제출 저장소

[00skgun/codessey_advance](https://github.com/00skgun/codessey_advance)
