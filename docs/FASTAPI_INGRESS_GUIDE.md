# INFRIUM AI② FastAPI 수신 서버 안내

## 실행

프로젝트 루트에서 가상환경을 활성화한 뒤 실행한다.

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8002
```

개발 환경에서만 `0.0.0.0` 바인딩을 사용한다. 인증 없는 서버를 공용 인터넷에 노출하면 안 된다. 현재는 단일 worker만 지원한다.

## AI① 요청

`POST /api/ai1/windows`로 `Content-Type: application/json` 요청을 보낸다. 최상위 필수 필드는 `schema_version`, `message_type=ai1_window`, `bridge_type`, `experiment_id`, `window_id`, `window_start`, `window_start_us`, `window_end_us`, `raw`, `ai1`이다.

`ai1.features`에는 22개 Model C Feature가 모두 finite number로 있어야 한다. Model C는 RAW 센서 배열을 직접 입력으로 사용하지 않는다. AI②는 `window_end_us`를 UTC event timestamp로 변환하고 `received_at`과 구분한다.

## 응답

성공 응답에는 XGBoost 구조 상태 분류, 확률, 정상 기준 변화량, AI① 결과 보존값, data quality가 포함된다. RiskScore 검증 전이므로 항상 다음을 유지한다.

```json
{
  "risk_score": null,
  "risk_level": null,
  "risk_status": "PENDING_VALIDATION"
}
```

`D1`, `D2`, `D3`는 손상 유형이며 위험도 단계가 아니다. `warming_up=true`이면 `ai1_status`는 `PENDING_WARMUP`이며 AI① prediction을 정상 판정 근거로 사용하지 않는다.

## 중복과 오류

- 동일 `bridge_type + experiment_id + window_id`·동일 내용: HTTP 200, cached duplicate
- 동일 키·다른 내용: HTTP 409 `WINDOW_CONFLICT`
- 동일 `experiment_id + window_id`의 다른 bridge: HTTP 409
- validation 실패: HTTP 422
- 요청 크기 초과: HTTP 413. 기본 한도 1 MiB, `AI2_MAX_REQUEST_BYTES`로 조정 가능
- 모델/추론 불가: HTTP 503

현재 메모리 캐시는 재시작 시 소실되고 다중 worker 사이에서 공유되지 않는다. 202 비동기 접수·Django POST·영속 작업 큐·인증·운영 배포는 이번 구현 범위에 포함되지 않는다.
