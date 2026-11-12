# next-station-ai

다음역의 AI 파트 레포입니다. 강의 노트로 요약 카드, 퀴즈, 단어 카드를 만드는 데 쓰는 LLM 출력 스키마, 프롬프트, 예시 데이터를 관리합니다.

LLM 호출 코드는 백엔드 레포에 있고, 이 레포에서는 스키마·프롬프트·검증 규칙을 정해서 넘깁니다.

## 파일

| 경로 | 설명 |
| --- | --- |
| `schema/llm-output.schema.json` | Gemini 응답 스키마. 모델이 이 형식으로 JSON을 출력 |
| `samples/llm-output.example.json` | 위 스키마를 따르는 LLM 출력 예시 |
| `samples/api-response.example.json` | 백엔드가 프론트에 내려줄 응답 예시 (제안안) |

## 데이터 flow

```
강의 노트 (최대 약 5,000자)
  → Gemini (gemini-3.1-flash-lite) + llm-output.schema.json
  → LLM 출력 (llm-output)
  → 백엔드 가공: 검증, id 부여, 블록 구성, 예상 소요 시간 계산
  → API 응답 (api-response)
  → 프론트 (IndexedDB 저장)
```

예상 소요 시간은 LLM이 아니라 백엔드에서 규칙으로 계산합니다. 초기값은 읽기 약 7자/초, TTS 약 5자/초, 퀴즈 30초/문항이고 실측 후 조정할 예정입니다.
