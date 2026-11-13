"""samples/notes/*.txt 로 Gemini를 호출하고 결과를 검증한다.

사용법 (레포 루트에서):
    pip install -r eval/requirements.txt
    python eval/run_eval.py                      # 전체 노트 1회씩
    python eval/run_eval.py --runs 3             # 노트당 3회 (응답 시간 편차 확인)
    python eval/run_eval.py --thinking LOW       # thinkingLevel 지정 (기본값은 모델 기본)
    python eval/run_eval.py --dry-run            # 호출 없이 첫 노트의 요청 본문만 출력
    python eval/run_eval.py --prompt 다른프롬프트.txt  # 다른 시스템 프롬프트와 비교

요청 본문(사용자 메시지 형식, generationConfig)은 백엔드 LearningContentService 와 똑같이 만든다.

GEMINI_API_KEY 는 레포 루트의 .env 에서 읽는다.
결과는 eval/results/<시각>/ 에 저장된다.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "llm-output.schema.json"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system.md"
NOTES_DIR = ROOT / "samples" / "notes"
RESULTS_DIR = ROOT / "eval" / "results"

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# 백엔드 LearningContentService 와 같은 형식
USER_TEMPLATE = "다음 강의 노트로 학습 콘텐츠를 만들어 주세요.\n\n<note>\n{note}\n</note>"
NOTE_MAX_CHARS = 5000
LATENCY_TARGET_SEC = 25.0
HTTP_TIMEOUT_SEC = 90  # 25초 넘는 경우도 측정하려고 넉넉하게

# ttsScript 에 있으면 안 되는 문자 (마침표, 쉼표, 물음표 외 기호)
TTS_FORBIDDEN = re.compile(r"[()\[\]{}<>/:;\"'“”‘’→←↔·•*#_~|=+\-]")
LATIN = re.compile(r"[A-Za-z]")


def load_env(path):
    if not path.exists():
        return
    data = path.read_bytes()
    # PowerShell 5.1 의 `echo ... > .env` 는 UTF-16 으로 저장한다
    text = data.decode("utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else data.decode("utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_user_message(note):
    return USER_TEMPLATE.format(note=note)


def build_request(system_prompt, note, schema, thinking_level):
    generation_config = {
        "responseMimeType": "application/json",
        "responseJsonSchema": schema,
    }
    if thinking_level:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_message(note)}]}],
        "generationConfig": generation_config,
    }


RETRYABLE_HTTP = (429, 503)


def call_gemini(model, api_key, body, max_retries):
    """반환: (raw, 마지막 시도의 응답 시간, 429/503 재시도 횟수).
    백엔드는 HTTP 오류 시 재시도하지 않으므로, 재시도 횟수 = 운영이었다면 502 가 났을 횟수."""
    req = urllib.request.Request(
        API_URL.format(model=model),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    retries = 0
    while True:
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = {"httpError": e.code, "body": e.read().decode("utf-8", "replace")}
        except (urllib.error.URLError, TimeoutError) as e:
            raw = {"networkError": str(e)}
        elapsed = time.perf_counter() - start
        if raw.get("httpError") in RETRYABLE_HTTP and retries < max_retries:
            retries += 1
            time.sleep(5 * retries)
            continue
        return raw, elapsed, retries


def extract_output(raw):
    """응답에서 (출력 텍스트, finishReason) 을 꺼낸다. thought 파트는 제외."""
    candidates = raw.get("candidates") or []
    if not candidates:
        return None, None
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return text, cand.get("finishReason")


def normalize(s):
    return re.sub(r"\s+", " ", s).strip()


def check_lengths(schema, data, path, errors):
    """스키마 description 의 'N자 이내' 를 읽어 글자 수를 검사한다.
    Gemini 가 maxLength 를 강제하지 않으므로 description 에만 적혀 있는 제한이다."""
    t = schema.get("type")
    if t == "object" and isinstance(data, dict):
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                check_lengths(sub, data[key], f"{path}.{key}", errors)
    elif t == "array" and isinstance(data, list):
        for i, item in enumerate(data):
            check_lengths(schema.get("items", {}), item, f"{path}[{i}]", errors)
    elif t == "string" and isinstance(data, str):
        m = re.search(r"(\d+)자 이내", schema.get("description", ""))
        if m and len(data) > int(m.group(1)):
            errors.append(f"길이 초과 {path}: {len(data)}자 > {m.group(1)}자")


def blank(v):
    return not isinstance(v, str) or not v.strip()


def backend_validate(output):
    """백엔드 LlmOutputValidator 와 같은 규칙. 여기서 실패하면 운영에서는 재요청 후 502."""
    errs = []
    if not isinstance(output, dict):
        return ["출력이 객체가 아님"]
    if blank(output.get("title")):
        errs.append("title 없음")

    topics = output.get("topics")
    if not isinstance(topics, list) or not 2 <= len(topics) <= 4:
        return errs + ["topics는 2~4개여야 함"]
    for i, t in enumerate(topics):
        w = f"topics[{i}]"
        if not isinstance(t, dict):
            errs.append(f"{w} 비어 있음")
            continue
        if blank(t.get("title")):
            errs.append(f"{w}.title 없음")
        if blank(t.get("ttsScript")):
            errs.append(f"{w}.ttsScript 없음")
        cards = t.get("summaryCards")
        if not isinstance(cards, list) or not 1 <= len(cards) <= 2:
            errs.append(f"{w}.summaryCards는 1~2개여야 함")
        elif any(not isinstance(c, dict) or blank(c.get("heading")) or blank(c.get("body")) for c in cards):
            errs.append(f"{w}.summaryCards에 빈 카드 있음")
        terms = t.get("terms")
        if not isinstance(terms, list) or not 1 <= len(terms) <= 3:
            errs.append(f"{w}.terms는 1~3개여야 함")
        elif any(not isinstance(x, dict) or blank(x.get("term")) or blank(x.get("definition")) for x in terms):
            errs.append(f"{w}.terms에 빈 용어 있음")

    quizzes = output.get("quizzes")
    if not isinstance(quizzes, list) or len(quizzes) != 5:
        return errs + ["quizzes는 정확히 5개여야 함"]
    for i, q in enumerate(quizzes):
        w = f"quizzes[{i}]"
        if not isinstance(q, dict):
            errs.append(f"{w} 비어 있음")
            continue
        ti = q.get("topicIndex")
        if not isinstance(ti, int) or not 0 <= ti < len(topics):
            errs.append(f"{w}.topicIndex={ti} 범위 밖")
        if blank(q.get("question")) or blank(q.get("explanation")) or blank(q.get("evidence")):
            errs.append(f"{w} 질문/해설/근거 중 빈 값 있음")
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) != 4:
            errs.append(f"{w}.options는 4개여야 함")
        else:
            if any(blank(o) for o in opts):
                errs.append(f"{w}.options에 빈 보기 있음")
            if len(set(opts)) != 4:
                errs.append(f"{w}.options에 중복 보기 있음")
        ai = q.get("answerIndex")
        if not isinstance(ai, int) or not 0 <= ai <= 3:
            errs.append(f"{w}.answerIndex={ai}")
    return errs


def check_output(output, note, schema):
    """반환: dict(schema_errors, backend_errors, length_errors, evidence_ok, evidence_total, warnings)"""
    result = {
        "schema_errors": [],
        "backend_errors": backend_validate(output),
        "length_errors": [],
        "evidence_ok": 0,
        "evidence_total": 0,
        "warnings": [],
    }

    for e in jsonschema.Draft202012Validator(schema).iter_errors(output):
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        msg = e.message if len(e.message) <= 120 else f"{e.validator}={e.validator_value} 위반"
        result["schema_errors"].append(f"{loc}: {msg}")

    if not isinstance(output, dict):
        return result

    warn = result["warnings"]
    topics = output.get("topics") if isinstance(output.get("topics"), list) else []
    quizzes = output.get("quizzes") if isinstance(output.get("quizzes"), list) else []

    for i, q in enumerate(quizzes):
        ev = q.get("evidence") if isinstance(q, dict) else None
        if isinstance(ev, str):
            result["evidence_total"] += 1
            if ev.strip() and normalize(ev) in normalize(note):
                result["evidence_ok"] += 1
            else:
                warn.append(f"quizzes[{i}].evidence 원문 불일치: {ev}")
        if isinstance(q, dict) and isinstance(q.get("options"), list):
            stripped = [o.strip() for o in q["options"] if isinstance(o, str)]
            if len(set(stripped)) != len(stripped):
                warn.append(f"quizzes[{i}].options 공백만 다른 중복 보기")

    # 백엔드는 글자 수를 검사하지 않으므로 여기서만 확인한다
    check_lengths(schema, output, "$", result["length_errors"])

    # 경고: 바로 실패는 아니지만 품질 확인이 필요한 것
    covered = {q.get("topicIndex") for q in quizzes if isinstance(q, dict)}
    for i in range(len(topics)):
        if i not in covered:
            warn.append(f"topics[{i}] 에 퀴즈 없음")

    answers = Counter(q.get("answerIndex") for q in quizzes if isinstance(q, dict))
    if answers and max(answers.values()) >= 3:
        warn.append(f"정답 위치 쏠림: {dict(sorted(answers.items()))}")

    for i, t in enumerate(topics):
        tts = t.get("ttsScript", "") if isinstance(t, dict) else ""
        bad = sorted(set(TTS_FORBIDDEN.findall(tts)))
        if bad:
            warn.append(f"topics[{i}].ttsScript 기호 포함: {' '.join(bad)}")
        if LATIN.search(tts):
            warn.append(f"topics[{i}].ttsScript 영문 포함")

    if len(note) > NOTE_MAX_CHARS:
        warn.append(f"노트 길이 {len(note)}자 > {NOTE_MAX_CHARS}자")

    return result


def run_one(model, api_key, system_prompt, schema, note_path, thinking_level, out_dir, run_idx, max_retries):
    note = note_path.read_text(encoding="utf-8")
    body = build_request(system_prompt, note, schema, thinking_level)
    raw, elapsed, retries = call_gemini(model, api_key, body, max_retries)

    name = note_path.stem + (f"-run{run_idx}" if run_idx else "")
    (out_dir / f"{name}.raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    usage = raw.get("usageMetadata") or {}
    row = {
        "note": name,
        "chars": len(note),
        "elapsed": elapsed,
        "retries": retries,
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "thought_tokens": usage.get("thoughtsTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
        "finish": None,
        "errors": [],
        "checks": None,
    }

    if "httpError" in raw or "networkError" in raw:
        row["errors"].append(f"호출 실패: {raw.get('httpError') or raw.get('networkError')}")
        return row

    text, finish = extract_output(raw)
    row["finish"] = finish
    if finish != "STOP":
        row["errors"].append(f"finishReason={finish}")
    if not text:
        block = (raw.get("promptFeedback") or {}).get("blockReason")
        row["errors"].append(f"출력 없음 (blockReason={block})")
        return row

    try:
        output = json.loads(text)
    except json.JSONDecodeError as e:
        row["errors"].append(f"JSON 파싱 실패: {e}")
        return row

    (out_dir / f"{name}.output.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    row["checks"] = check_output(output, note, schema)
    if elapsed > LATENCY_TARGET_SEC:
        row["checks"]["warnings"].append(f"응답 {elapsed:.1f}초 > {LATENCY_TARGET_SEC:.0f}초")
    return row


def fmt(v):
    return "-" if v is None else str(v)


def pass_fail(errors):
    return "통과" if not errors else f"실패 {len(errors)}"


def summarize(rows, model, thinking_level, prompt_path):
    lines = [
        "# eval 결과",
        "",
        f"- 시각: {datetime.now().isoformat(timespec='seconds')}",
        f"- 모델: {model}",
        f"- thinkingLevel: {thinking_level or '설정 안 함 (백엔드와 같음)'}",
        f"- 시스템 프롬프트: {prompt_path}",
        "- 백엔드 검증: 백엔드 LlmOutputValidator 와 같은 규칙. 실패하면 운영에서는 1회 재요청 후 502",
        "- 글자 수: 스키마 description 기준. 백엔드는 검사하지 않음",
        "- 재시도: Gemini 429/503 재시도 횟수. 백엔드는 HTTP 오류 시 재시도 없이 502 를 돌려줌",
        "",
        "| 노트 | 글자 수 | 시간(초) | 재시도 | 입력 토큰 | 출력 토큰 | 생각 토큰 | finish | 스키마 | 백엔드 검증 | 글자 수 | evidence 일치 | 경고 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for r in rows:
        c = r["checks"]
        if c is None:
            cols = ["-", "실패" if r["errors"] else "-", "-", "-", "-"]
        else:
            cols = [
                pass_fail(c["schema_errors"]),
                pass_fail(c["backend_errors"]),
                pass_fail(c["length_errors"]),
                f"{c['evidence_ok']}/{c['evidence_total']}",
                str(len(c["warnings"])),
            ]
        lines.append(
            f"| {r['note']} | {r['chars']} | {r['elapsed']:.1f} | {r['retries']} | {fmt(r['prompt_tokens'])} | "
            f"{fmt(r['output_tokens'])} | {fmt(r['thought_tokens'])} | {fmt(r['finish'])} | "
            + " | ".join(cols) + " |"
        )

    lines += ["", "## 상세", ""]
    for r in rows:
        c = r["checks"] or {}
        items = (
            [f"오류: {e}" for e in r["errors"]]
            + [f"스키마: {e}" for e in c.get("schema_errors", [])]
            + [f"백엔드 검증: {e}" for e in c.get("backend_errors", [])]
            + [f"글자 수: {e}" for e in c.get("length_errors", [])]
            + [f"경고: {e}" for e in c.get("warnings", [])]
        )
        lines.append(f"### {r['note']}")
        lines += [f"- {i}" for i in items] if items else ["- 문제 없음"]
        lines.append("")
    return "\n".join(lines)


def has_failure(row):
    c = row["checks"]
    return bool(
        row["errors"] or c is None
        or c["schema_errors"] or c["backend_errors"] or c["length_errors"]
    )


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Gemini 출력 평가")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--runs", type=int, default=1, help="노트당 반복 횟수")
    parser.add_argument("--thinking", choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--notes", default="*.txt", help="samples/notes 안에서 고를 파일 패턴")
    parser.add_argument("--prompt", type=Path, default=SYSTEM_PROMPT_PATH,
                        help="시스템 프롬프트 파일 (기본: prompts/system.md)")
    parser.add_argument("--retries", type=int, default=3, help="Gemini 429/503 재시도 횟수")
    parser.add_argument("--dry-run", action="store_true", help="호출 없이 요청 본문만 출력")
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    system_prompt = args.prompt.read_text(encoding="utf-8")
    notes = sorted(NOTES_DIR.glob(args.notes))
    if not notes:
        sys.exit(f"노트 없음: {NOTES_DIR / args.notes}")

    if args.dry_run:
        body = build_request(system_prompt, notes[0].read_text(encoding="utf-8"), schema, args.thinking)
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return

    load_env(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY 가 없습니다. 레포 루트 .env 에 GEMINI_API_KEY=... 를 넣어 주세요.")

    out_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for note_path in notes:
        for run in range(1, args.runs + 1):
            label = note_path.stem + (f" (run {run})" if args.runs > 1 else "")
            print(f"호출 중: {label}", flush=True)
            rows.append(
                run_one(
                    args.model, api_key, system_prompt, schema, note_path,
                    args.thinking, out_dir, run if args.runs > 1 else 0, args.retries,
                )
            )

    try:
        prompt_label = args.prompt.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        prompt_label = args.prompt.name
    summary = summarize(rows, args.model, args.thinking, prompt_label)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"결과 저장: {out_dir.relative_to(ROOT)}")
    sys.exit(1 if any(has_failure(r) for r in rows) else 0)


if __name__ == "__main__":
    main()
