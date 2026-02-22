# Local CSV Analyzer (Streamlit)

로컬에서 단독 실행하는 CSV 분석 앱입니다. 목적은 **재현 가능한 자동 EDA + 지도/비지도 모델링 + LLM 해석 리포트**입니다.

## 기능 요약
- 강건한 CSV 로딩
  - 인코딩 순차 시도: `utf-8` → `utf-8-sig` → `cp949`
  - 실패 시 `sep=None, engine='python'`로 fallback
  - 결측 문자열(`""`, `NA`, `N/A`, `null`, `-`, `?` 등) 자동 처리
  - `max_rows` 초과 시 deterministic 샘플링(고정 seed)
- 자동 스키마 추정(룰 기반)
  - numeric / datetime / categorical
  - `id_candidate`, `long_text` 자동 감지 후 모델 입력 기본 제외
- 지도학습
  - 누수 방지: `Pipeline + ColumnTransformer`
  - numeric: median imputation
  - categorical: 컬럼 특성 기반 자동 인코딩
    - low-cardinality: one-hot
    - high-cardinality: frequency encoding
  - 옵션: IQR clipping
  - 모델: RandomForest(항상), XGBoost/LightGBM(설치 시)
  - 지표: 분류(`accuracy`, `f1_macro`, 가능 시 `roc_auc`), 회귀(`rmse`, `mae`, `r2`)
- 비지도학습
  - numeric 중심
  - `StandardScaler → PCA(2D) → KMeans`
  - 지표: `silhouette`, `davies_bouldin`, `calinski_harabasz`
  - PCA scatter + cluster profile(mean/median)
- 시각화 한글 지원
  - 실행 시 Matplotlib 한글 폰트를 자동 탐지/적용(`Malgun Gothic` 우선)
  - 가능한 경우 그래프 한글 라벨/제목이 깨지지 않게 표시
- LLM 리포트
  - 분석 버튼 이후에만 생성 가능
  - 입력: `report_data` JSON + `df.head(10)`
  - 강제 규칙: **JSON에 있는 숫자만 인용**

## 설치
Python 3.11+ 권장

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## 실행
```bash
streamlit run app.py
```

## 환경변수
- `OPENAI_API_KEY` (선택)
  - 없으면 LLM 리포트 기능만 비활성화되고 앱은 정상 동작
- `OPENAI_MODEL` (선택)
  - 기본값: `gpt-4.1-mini`

Windows(PowerShell) 예시:
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
streamlit run app.py
```

## 사용 방법
1. 사이드바에서 CSV 업로드
2. 분석 모드 선택(지도/비지도)
3. (지도학습) 타겟 컬럼 선택
4. 옵션 설정
   - `max_rows`, `test_size`, `random_seed`
   - 이상치 처리: 없음 / IQR 클리핑
   - 모델링: 빠름 / 느림(CV 5-fold)
5. `분석 실행`
6. 필요 시 `LLM 리포트 생성`

## 출력 탭
- `Overview`: 데이터 크기/타입 요약 + 상위 20행
- `EDA`: 결측률, 수치형 분포, 범주형 빈도, 상관 히트맵
  - 수치형 표준화 값 heatmap 포함
- `Preprocessing Log`: 컬럼별 전처리/제외 로그
- `Modeling`: 지도 또는 비지도 결과 테이블/그래프
- `LLM Report`: 마크다운 리포트

## 제한사항
- MVP 기준으로 datetime은 모델 입력에서 기본 제외
- 비지도는 numeric 중심(고차원 sparse categorical 미포함)
- XGBoost/LightGBM은 기본 의존성으로 설치됨(환경 이슈로 import 실패 시 skip 처리)
- LLM 리포트는 해석/설명 용도이며 계산 자체를 대체하지 않음
