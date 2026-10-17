# INFRIUM AI② RiskScore 기술 설계 v1

## 1. 목적과 범위

AI②는 AI①이 제공하는 HW 센서 기반 Window JSON을 받아 다음 결과를 별도로 제공한다.

1. 구조 상태 분류: `NORMAL`, `D1`, `D2`, `D3`
2. 정상 기준 대비 진동 Feature 변화량
3. 설명 가능한 `RiskScore` 후보값(0~100)
4. 위험도 상태: 정상, 주의, 경고, 위험

`D1`, `D2`, `D3`는 손상 **유형**이며 위험도의 순서가 아니다. XGBoost 분류 확률도 구조 안전성 확률이나 위험도 점수가 아니다. RiskScore는 XGBoost와 분리된 규칙·기준 기반 모듈에서 계산한다.

이 문서는 설계 가설이다. 가중치, 계수, RiskScore 산식의 운영값, 위험도 임계값은 확정하지 않는다. 현재 더미 CSV 결과는 실제 HW 안전성 검증 결과가 아니다.

## 2. 현재 근거와 핵심 제약

- 구조 상태 분류의 진동 중심 기준선은 Model C(22개 진동 Feature)다.
- Model D는 `temperature_c`에 과도하게 의존했으므로 RiskScore 근거로 사용하지 않는다.
- BOX/TRUSS 정상 기준은 AI② Train의 NORMAL 실험으로 만들어졌으나, 모든 하중조건이 `INSUFFICIENT_EXPERIMENTS`다.
- AI①은 같은 하중조건의 정상과 비교한 이상 정보를 제공한다. AI① 값과 AI② 진동 변화량은 일부 같은 원천 Feature를 반영하므로 무비판적으로 합산하면 이중 반영 위험이 있다.
- 현재 분석은 Test 데이터를 사용하지 않았다.

## 3. 입력 데이터 정의

### 3.1 AI① → AI② 입력 계약 제안

```json
{
  "schema_version": "ai1-ai2-v1",
  "message_id": "uuid-or-source-id",
  "bridge_type": "BOX",
  "window_id": "BOX_000123",
  "timestamp": "2026-10-15T12:30:00+09:00",
  "features": {
    "acc01_rms_x": 0.0,
    "...22 vibration features...": 0.0
  },
  "environment": {
    "load_level_kg": 1.0,
    "load_position": "C",
    "temperature_c": 24.5
  },
  "ai1_result": {
    "model_version": "required",
    "anomaly_score": 0.0,
    "anomaly_score_threshold": 0.0,
    "max_abs_z": 0.0,
    "z_threshold": 0.0,
    "decision_ratio": 0.0,
    "raw_prediction": "Normal",
    "prediction": "Normal",
    "sensor_status": {}
  }
}
```

필수 운영 메타데이터는 `message_id`, 모델/센서 버전, 센서 상태, 생성 시각이다. AI②는 `bridge_type + window_id`만으로 중복을 판단하지 않고 `message_id` 또는 센서·Window 시각을 함께 사용한다.

### 3.2 기준 조회

정상 기준은 `bridge_type`과 하중조건으로 조회한다.

| 입력 환경 | 기준 조회 키 |
|---|---|
| 0kg, 빈 위치 | `0kg-NONE` |
| 1kg, C | `1kg-C` |
| 1kg, E | `1kg-E` |
| 2kg, C | `2kg-C` |
| 2kg, E | `2kg-E` |

0kg가 아닌데 위치가 없으면 입력 오류다. 없는 조건의 기준을 다른 조건으로 대체하지 않는다. `temperature_c`는 기록·품질 점검·향후 층화 변수지만, 현 단계에서는 위험 점수의 직접 입력으로 확정하지 않는다.

## 4. 진동 변화 지표 후보

모든 22개 Feature를 단순 합산하지 않는다. 같은 센서·축·추출 방식에서 강하게 중복될 수 있으므로 Feature 그룹 내 요약, 센서 품질 검증, 실제 HW 반복 검증이 선행돼야 한다.

| 그룹 | 물리적 의미 | 정상 대비 변화 해석 | 실제 HW 가능성·주의 | 중복 가능성 |
|---|---|---|---|---|
| RMS 6개 | Window 내 진동 에너지/크기 | 증가·감소 모두 관찰 대상이며 부호 자체를 위험 방향으로 고정하지 않음 | 타격 크기, 장착 상태, 노이즈, 하중에 민감 | 동일 센서의 P2P와 중복 가능 |
| P2P 6개 | Window 내 진폭 범위 | 충격 응답 진폭 변화 후보 | 순간 잡음·클리핑·충격 재현성 점검 필요 | RMS·상위 peak Feature와 중복 가능 |
| 주요 진동수 6개 | 지배 주파수, 동특성 변화 후보 | 이동 방향은 모드와 경계조건에 따라 달라 고정하지 않음 | FFT 길이, 샘플링률, 주파수 분해능, aliasing 관리 필요 | x/y/z 및 두 센서의 모드 정보 중복 가능 |
| 감쇠 2개 | 진동 소산 속도 | 감쇠 증가·감소 모두 구조·접촉·추정오차 원인이 가능 | 충격 시작점, fitting 구간, SNR에 민감 | 같은 모드 주파수·RMS와 상관 가능 |
| 센서 간 응답 2개 | L/4:L/2 진폭비, 비틀림 관련 비 | 형상·비대칭 변화 후보 | 분모가 작을 때 비율 불안정, 두 센서 동기화 필요 | 개별 RMS 및 축 Feature와 중복 가능 |

### 4.1 현재 더미 데이터에서 우선 검토할 후보

실험 단위 Train 분석에서 BOX는 `acc01_damping_z`, `acc02_damping_z`, 일부 주요 진동수 변화가, TRUSS는 두 감쇠 Feature, `cross_amp_ratio_L4_L2_z`, Y축 RMS/P2P 계열이 후보로 나타났다. 이는 후보 우선순위일 뿐 실제 위험 기여도나 공인 안전 지표가 아니다.

후보 선정 원칙:

1. 각 물리 그룹에서 중복성이 낮고 센서 품질이 확인된 Feature를 우선 검토한다.
2. 한 그룹의 다수 Feature가 같은 현상을 반영하면 그룹 요약값 또는 대표 Feature를 사용한다.
3. 손상 유형별로 Feature 방향이 다르면 절대 변화량 또는 양·음 변화의 분리 표시를 검토한다.
4. 실제 HW에서 반복성, 환경 강건성, 센서 교체 후 재현성을 검증한다.

## 5. 정규화 방식 비교

| 방식 | 정의 | 장점 | 기준 부족 시 문제 |
|---|---|---|---|
| 상대 변화율 | `(x - mean) / abs(mean)` | 단위가 달라도 변화 비율을 직관적으로 비교 가능 | 평균이 0 또는 매우 작으면 불안정, 작은 정상 표본의 평균 오차가 증폭 |
| 표준편차 기반 | `(x - mean) / std` | 정상 분포 폭을 반영, 방향 보존 가능 | 현재 조건별 실험 수가 1~2개여서 `std`가 대표성이 없거나 추정 불가 |
| IQR 기반 | `(x - median) / IQR` 또는 robust z-score | 극단값에 표준편차보다 강건 | 작은 표본에서 IQR이 0 또는 불안정, Window 상관성이 표본 수를 과대평가 |
| 분위수 기반 | 정상 기준의 empirical percentile | 비선형 분포·단위 차이에 강건 | 충분한 독립 NORMAL 반복이 없으면 분위수 위치를 신뢰할 수 없음 |

운영 전 권고는 `기준 없음/부족`을 점수로 보정하지 말고 데이터 신뢰성 상태로 먼저 반환하는 것이다. 현재는 어떤 정규화 방식도 운영값으로 확정할 근거가 없다.

## 6. AI① 결과 활용 방안

| 입력 | 확인된 의미 | 잠재적 역할 | 이중 반영 위험 및 제한 |
|---|---|---|---|
| `anomaly_score` | Isolation Forest 점수, 0~1, 클수록 이상 | 이상 증거의 연속값 후보 | AI② 진동 Feature와 공통 원천이므로 변화량 지수와 직접 합산 전 상관·중복 검증 필요 |
| `decision_ratio` | `max(score/threshold, abs(z)/z_threshold)`, 1 초과 이상 | 조건별 기준 초과 정도의 후보 | 파생 통계이며 score와 max z를 중복 반영할 수 있음. AI① 문서·버전·threshold를 함께 저장해야 함 |
| `raw_prediction` | Window 단위 Normal/Anomaly | 이산형 보조 플래그 | 연속점수보다 정보량이 낮고 단독 RiskScore 입력으로 부적합 |
| `prediction` | 최근 3 Window 중 2개 이상 이상이면 Anomaly | 표시/알림의 지속성 신호 | 이미 시간 집계된 결과이므로 AI②가 다시 같은 지속성 규칙을 적용하면 이중 시간 평활 위험 |

원칙은 한 Window에서 `anomaly_score`, `decision_ratio`, `raw_prediction`, `prediction`을 모두 독립 증거처럼 더하지 않는 것이다. AI① 계열은 하나의 대표 연속지표와 별도의 지속성 상태로 분리해 검증한다. 범위·방향·버전이 확인되지 않은 새 AI① 입력은 점수 합산에 넣지 않는다.

## 7. XGBoost Model C 활용 방안

Model C의 클래스 순서는 고정한다.

```text
[NORMAL, D1, D2, D3]
```

활용 가능한 출력은 클래스별 확률, 최대 확률, 예측 클래스, 모델/feature schema 버전이다. RiskScore에서는 다음과 같이 제한적으로 사용한다.

- 구조 상태 **가설**과 분류 불확실성 표시
- 운영자 화면에서 D1/D2/D3 후보를 병렬 표시
- 다른 독립 증거가 있을 때 조사 우선순위의 보조 신호

사용하면 안 되는 해석:

- `P(D3) > P(D1)`을 더 위험하다고 해석
- `P(NORMAL)`을 안전 확률로 해석
- 높은 최대 확률을 위험도 상승 근거로 단독 사용

Validation에서 Model C는 BOX D3→NORMAL, TRUSS D1↔D2 및 D3→NORMAL 혼동 사례를 보였다. 그러므로 Model C 확률은 calibration, HW 재검증, 데이터 신뢰성 상태 없이는 위험 점수의 직접 승수로 쓰지 않는다.

## 8. RiskScore 산식 후보

아래 후보는 설명 가능한 연구용 구조이며, `w`, `T`, `f`, `g`는 실제 HW 검증과 합의 후에만 정한다. 현재 구현하지 않는다.

### 후보 A: 독립 증거 게이트형 합성 지수

입력:

- `V`: 하중조건 일치 정상 기준 대비 진동 변화 지수. 그룹별 대표 변화량을 robust normalization한 뒤 그룹 중복을 줄여 요약한다.
- `A`: AI①의 검증된 대표 이상 연속지표. 초기 검토 대상은 `decision_ratio` 또는 `anomaly_score` 중 하나이며 둘을 동시 합산하지 않는다.
- `Q`: 데이터 신뢰성 상태. 수치 가중치가 아니라 계산 허용/보류 게이트다.

개념식:

```text
if Q != VALID:
    RiskScore = UNAVAILABLE 또는 PROVISIONAL
else:
    V* = f(V)                  # 0~1로 보정된 진동 변화 증거
    A* = g(A)                  # 0~1로 보정된 AI① 이상 증거
    RiskScore = 100 × clip(wv × V* + wa × A*, 0, 1)
```

장점:

- 진동 변화와 AI① 이상 탐지를 명시적으로 분리한다.
- `Q`가 부족하면 정상으로 오인하지 않고 보류할 수 있다.
- 각 성분을 운영자에게 설명할 수 있다.

한계:

- `V`와 `A`가 동일 진동 Feature에 의존할 수 있어 가중 합산 전 상관·중복 검증이 필요하다.
- `wv`, `wa`, `f`, `g`는 현 데이터로 확정할 수 없다.
- 현재 모든 정상 기준이 부족하므로 운영 Score를 산출하면 안 된다.

HW 검증:

- 조건별 반복 NORMAL, 손상·환경 독립 실험, 센서 결함 시나리오에서 score monotonicity와 오경보율을 검증한다.

### 후보 B: 진동 변화 우선 + AI① 확인형 지수

입력:

- `V`: 그룹화된 진동 변화 증거
- `A`: AI① 대표 이상지표
- `P`: AI① 지속성 상태(`prediction`) 또는 AI② 측 시간적 일관성 상태. 둘 중 하나의 시간 집계만 사용한다.

개념식:

```text
if Q != VALID:
    RiskScore = UNAVAILABLE 또는 PROVISIONAL
else:
    base = h(V)                # 진동 변화만으로 만든 0~100 후보값
    confirmation = k(A, P)     # 확인 신호; 상승만 허용하거나 표시 상태만 갱신
    RiskScore = combine(base, confirmation)
```

`combine`은 단순 합이 아니라 다음 중 하나를 검증 후 선택한다.

- 확인 신호가 없으면 `base` 유지, 있으면 별도 검토 상태로 승격
- 두 증거가 모두 지속될 때만 score 갱신
- `max(base, confirmed_signal)` 같은 보수적 구조

장점:

- RiskScore의 주 근거를 물리 Feature 변화에 둔다.
- AI①을 같은 신호의 중복 가중치가 아니라 확인 신호로 취급할 수 있다.

한계:

- 확인 규칙과 시간 집계 창을 정해야 한다.
- AI① miss가 발생하면 진동 변화가 있어도 경보가 지연될 수 있다.
- `h`, `k`, `combine`에 대한 HW 검증이 필요하다.

### 후보 C: 점수 대신 증거 벡터 + 보류 우선 운영

초기 HW 검증 단계에서는 하나의 0~100 값을 즉시 운영하지 않고 다음을 함께 반환한다.

```text
evidence = {
  vibration_change: V,
  ai1_anomaly: A,
  model_c_hypothesis: [P(NORMAL), P(D1), P(D2), P(D3)],
  data_confidence: Q
}
```

충분한 HW 기반 label과 기준이 쌓인 뒤 후보 A 또는 B의 계수와 임계값을 calibration한다. 장점은 불확실한 데이터를 강제 수치화하지 않는다는 점이다. 한계는 초기 운영 화면이 단일 RiskScore를 요구하는 경우 정책 합의가 필요하다는 점이다.

### 추천

현재는 **후보 C를 1차 운영 인터페이스**, 충분한 정상 기준이 확보된 뒤 **후보 B를 우선 검증 후보**로 권고한다. 후보 B는 AI①·AI②의 중복 증거를 명시적으로 분리하기 쉽다. 후보 A는 독립성 검증과 calibration이 끝난 경우에만 비교 대상으로 유지한다.

## 9. 위험도 4단계와 데이터 신뢰성 상태

### 9.1 위험도 상태

위험도는 `정상 / 주의 / 경고 / 위험`의 네 단계로 표시할 수 있다. 단, 현재는 Score 임계값을 정하지 않는다. 각 단계는 손상 유형명이 아니라 검증된 RiskScore 구간과 지속성 정책으로만 정의해야 한다.

임계값 결정 절차:

1. HW 반복 NORMAL 데이터에서 허용 오경보율을 합의한다.
2. 각 손상 유형·severity·하중·온도 조건을 독립적으로 반복 측정한다.
3. Window와 실험 단위의 score 분포, 탐지 지연, 경보 지속성을 검증한다.
4. 운영자 검토 결과와 실제 점검 결과를 연결해 임계값을 고정한다.
5. 변경 시 모델·기준·임계값 버전을 기록하고 재검증한다.

### 9.2 위험도와 분리된 데이터 신뢰성 상태

| 상태 | 의미 | RiskScore 처리 |
|---|---|---|
| `VALID` | 조건 일치 기준, 필수 센서, 시각, 버전이 검증됨 | 계산 가능 |
| `PROVISIONAL_BASELINE` | 기준은 존재하나 독립 실험 수가 부족 | Score가 필요하면 잠정값으로 명시, 자동 위험도 확정 금지 |
| `INSUFFICIENT_BASELINE` | 현재 상태. 조건별 정상 기준 부족 | 정상/위험으로 강제 분류하지 않고 보류 또는 조사 필요 표시 |
| `OUT_OF_CONDITION` | 하중조건 기준 없음 | 대체 기준 없이 보류 |
| `SENSOR_DEGRADED` | 센서 품질·범위·동기화 이상 | Score 보류, 센서 점검 필요 |
| `MISSING_REQUIRED` | 필수 Feature·환경값 누락 | Score 보류 |
| `STALE_DATA` | 지연 허용범위를 넘는 메시지 | 실시간 상태 갱신 금지 |
| `DUPLICATE_WINDOW` | 재전송 또는 중복 Window | 재계산·재집계 금지 |
| `VERSION_MISMATCH` | feature/AI① 모델/schema 버전 불일치 | Score 보류 |

데이터 신뢰성 상태는 위험도보다 우선한다. `INSUFFICIENT_BASELINE`이 `NORMAL`이나 `위험`을 의미하지 않도록 UI와 API에서 별도 필드로 제공한다.

## 10. 실시간 처리 구조

### 10.1 Window 단위 처리

1. JSON schema, Feature 22개, 유한 숫자, 단위, bridge type을 검증한다.
2. `bridge_type + load_level_kg + load_position`으로 기준을 조회한다.
3. `feature_change`로 변화량을 계산한다.
4. Model C를 사용할 수 있는 schema일 때만 분류 확률과 손상 유형 가설을 조회한다.
5. AI① 연속점수와 지속성 신호를 분리한다.
6. 데이터 신뢰성 상태를 먼저 판정하고, 허용될 때만 선택된 RiskScore 후보를 계산한다.
7. 결과에 입력·기준·모델·정책 버전을 기록한다.

### 10.2 연속 Window 집계

- Window 단위 변화량/증거를 짧은 시간 창에서 집계하되, 창 길이와 집계함수는 HW 검증 후 확정한다.
- AI①의 `prediction`이 이미 최근 3 Window 지속성 규칙을 사용하므로 동일 규칙을 다시 Score에 더하지 않는다.
- 급변은 원본 Window 값과 센서 상태를 먼저 확인한다. 무조건 smoothing해 숨기지 않는다.
- 단계 변경은 별도 hysteresis 또는 최소 지속 조건을 검증 후 적용한다. 현재 값은 미정이다.
- 장기 추세와 즉시 경보를 구분해 저장한다.

### 10.3 메시지 지연·중복·결측

- 메시지는 `message_id`를 idempotency key로 저장한다.
- 지연 메시지는 수신 시각과 측정 시각을 분리하고, 허용 지연을 초과하면 `STALE_DATA`를 반환한다.
- 중복 Window는 재계산·집계에 넣지 않는다.
- 필수 센서 누락 시 부분 Score로 정상 판정을 하지 않는다. 지원되는 축소 Feature schema가 검증되기 전에는 `MISSING_REQUIRED`로 처리한다.
- BOX/TRUSS는 기준, 모델, 상태 저장소, time-series 집계를 독립적으로 관리한다.

### 10.4 AI② 출력 계약 제안

```json
{
  "schema_version": "ai2-risk-v1",
  "bridge_type": "BOX",
  "window_id": "BOX_000123",
  "timestamp": "2026-10-15T12:30:00+09:00",
  "damage_hypothesis": {
    "class_order": ["NORMAL", "D1", "D2", "D3"],
    "predicted_class": "D1",
    "probabilities": [0.05, 0.80, 0.10, 0.05],
    "model_version": "required"
  },
  "feature_change": {
    "baseline_status": "INSUFFICIENT_EXPERIMENTS",
    "baseline_source": "DUMMY_TRAIN_NORMAL",
    "load_condition": "1kg-C"
  },
  "risk": {
    "score": null,
    "level": null,
    "policy_version": null,
    "data_confidence": "INSUFFICIENT_BASELINE",
    "reason_codes": ["BASELINE_INSUFFICIENT"]
  }
}
```

`score: null`과 `level: null`은 계산 불가 또는 보류를 뜻하며 정상 상태가 아니다.

## 11. 실제 HW 검증 계획

1. BOX/TRUSS·하중조건별 독립 NORMAL 반복 실험을 최소 기준 이상으로 확보한다.
2. 온도, 하중, 손상 유형이 얽히지 않도록 실험 순서와 조건을 균형화한다.
3. 센서 장착, 교정, sampling rate, timestamp 동기화, 단위 변환을 기록한다.
4. 각 손상 유형과 severity를 반복 측정하고, 경계·혼합·센서결함 시나리오를 별도로 수집한다.
5. Window 단위와 실험 단위 평가를 분리한다. 같은 실험의 Window를 독립 검증 표본으로 계산하지 않는다.
6. Model C 분류 확률의 calibration, AI① 점수의 시간 안정성, 정상 기준의 drift를 검증한다.
7. 후보 A/B/C를 비교해 안전성, 오경보, 미탐, 지연, 설명 가능성을 평가한다.
8. 독립 holdout Test는 설계·선정이 완료된 뒤 한 번만 사용한다.

## 12. 미확정 사항과 의사결정 필요 항목

1. RiskScore의 사용 목적: 점검 우선순위, 알림, 자동 제어 중 무엇인지
2. 허용 가능한 오경보·미탐·지연 수준
3. 조건별 NORMAL 실험 최소 수와 기준 갱신 주기
4. Feature 그룹 대표값 및 중복 Feature 제거 정책
5. AI① 대표 입력 선택(`anomaly_score` 또는 `decision_ratio`)과 시간 집계 책임 주체
6. Model C 확률 calibration과 운영 중 재검증 정책
7. 데이터 신뢰성 상태별 UI, 알림, 사람 검토 절차
8. RiskScore 계수·가중치·단계 임계값의 HW 기반 확정 절차
9. JSON schema/versioning, 재전송, 보관 기간, 개인정보·운영 보안 정책

## 13. 비목표

이 문서는 Python 구현, RiskScore 산출, 위험도 단계 임계값 고정, XGBoost 재학습, Test 평가를 수행하지 않는다.
