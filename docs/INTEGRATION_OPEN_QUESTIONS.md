# AI①·AI②·백엔드 연동 미확정 사항

## AI① 팀 확인 필요

1. 실제 Feature 이름 22개가 계약 순서와 완전히 일치하는가?
2. RMS, P2P, dominant frequency, damping, ratio의 정확한 계산식·필터·Window 범위는 무엇인가?
3. Feature 단위, 가속도 센서 축·설치 위치·보정 방식은 무엇인가?
4. Window 길이와 이동 간격은 얼마이며 타격/이벤트와 어떻게 연결되는가?
5. 가속도 샘플링 주파수, 시간 동기화 기준, timestamp 생성 시점은 무엇인가?
6. `anomaly_score`, `decision_ratio`, threshold, z-score의 정의·범위·방향·모델 버전은 무엇인가?
7. `raw_prediction`과 `prediction`의 정확한 지속성 규칙과 초기 Window 정책은 무엇인가?
8. `model_version`, `sensor_status`, 센서 결측·고장·보정 상태의 JSON schema는 무엇인가?
9. 하중 위치 C/E/NONE 외의 값, 0kg에서 위치 표기, 하중 단계화 규칙은 무엇인가?

## 백엔드 팀 확인 필요

1. AI② 호출 방식은 HTTP request/response, queue, stream 중 무엇인가?
2. 메시지 ID, 재전송, retry, idempotency 저장 기간 정책은 무엇인가?
3. Window·분석 결과·기준/모델/policy 버전의 저장 schema와 보존 기간은 무엇인가?
4. `risk_score: null`, `risk_level: null`, `PENDING_VALIDATION`, 데이터 품질 상태를 UI에서 어떻게 표시하는가?
5. 중복·충돌·stale·out-of-order 메시지의 HTTP/queue acknowledgement 정책은 무엇인가?
6. 실시간 대시보드 갱신 주기, 최신 상태와 늦게 도착한 이력의 표시 규칙은 무엇인가?
7. BOX/TRUSS 상태 저장소를 단일 프로세스 메모리에서 Redis/DB로 옮길 운영 계획은 무엇인가?
8. 인증, 권한, 암호화, 감사 로그, 모니터링, 장애 복구 요구사항은 무엇인가?

## HTTP API 구현 전 공통 결정

- version negotiation과 schema migration 정책
- timestamp 허용 지연·미래 허용값의 운영 설정
- 실제 HW 기준에서 `VALID` baseline으로 승격하는 최소 반복실험 수
- RiskScore·4단계 위험도 정책의 HW 검증·승인·변경 관리
