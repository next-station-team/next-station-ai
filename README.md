# next-station-ai

다음역의 AI 파트 레포입니다. 강의 노트로 요약 카드, 퀴즈, 단어 카드를 만드는 데 쓰는 LLM 출력 스키마, 프롬프트, 예시 데이터를 관리합니다.

LLM 호출 코드는 백엔드 레포에 있고, 이 레포에서는 스키마·프롬프트·검증 규칙을 정해서 넘깁니다.

## 파일

| 경로 | 설명 |
| --- | --- |
| `schema/llm-output.schema.json` | Gemini 응답 스키마. 모델이 이 형식으로 JSON을 출력 |
| `samples/llm-output.example.json` | 위 스키마를 따르는 LLM 출력 예시 |
| `samples/api-response.example.json` | 백엔드가 프론트에 내려줄 응답 예시 (제안안) |
| `prompts/system.md` | 시스템 프롬프트. 백엔드 `resources/gemini/system-prompt.txt`에 반영 |
| `samples/notes/*.txt` | 테스트용 강의 노트 |
| `eval/run_eval.py` | 테스트 노트로 Gemini를 백엔드와 같은 형식으로 호출하고, 스키마·백엔드 검증·글자 수·evidence를 확인해 응답 시간, 토큰 수와 함께 표로 정리 |

## eval 실행

```
pip install -r eval/requirements.txt
echo GEMINI_API_KEY=발급받은키 > .env
python eval/run_eval.py
```

결과는 `eval/results/<시각>/`에 저장됩니다 (git에는 올라가지 않음). `--runs 3`으로 반복 측정, `--thinking LOW`로 thinkingLevel 지정, `--dry-run`으로 호출 없이 요청 본문만 확인할 수 있습니다.

## 데이터 flow

```
강의 노트 (최대 약 5,000자)
  → Gemini (gemini-3.1-flash-lite) + llm-output.schema.json
  → LLM 출력 (llm-output)
  → 백엔드 가공: 검증, id 부여, 블록 구성
  → API 응답 (api-response)
  → 프론트 (IndexedDB 저장)
```


