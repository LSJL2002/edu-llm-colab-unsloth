#!/usr/bin/env python3
"""
HuggingFace Hub에서 실제 한국어 교육 데이터셋을 받아 20시나리오 통합 포맷으로 변환.

설계 원칙
  - 손으로 만든 seed_train.jsonl 은 "예시(스타일 참고)"일 뿐 학습에 쓰지 않는다.
  - 실제 학습 데이터는 전부 HF Hub에서 받아온다. (사용자 요구)
  - 193k/105k 같은 대형도 streaming 으로 N개만 추출 → 전체 다운로드 불필요.

출력: data/hf_train.jsonl  (+ --val-ratio 시 hf_val.jsonl)
포맷: {"scenario_id", "instruction", "input", "output", "source"}

사용:
  pip install -q datasets
  python scripts/fetch_hf_datasets.py --per-source 8000 --val-ratio 0.05
  python scripts/fetch_hf_datasets.py --only socratic math   # 일부만
  python scripts/fetch_hf_datasets.py --list                 # 추천 목록만 출력
"""
import argparse
import json
import random
from pathlib import Path

# ── 추천 데이터셋 레지스트리 ────────────────────────────────────────────────
# 각 소스: key, hf_id, split, scenario_id, instruction(시스템 프롬프트),
#          map(row->(input,output)), license, note
SOURCES = {
    "general": {
        "hf_id": "beomi/KoAlpaca-v1.1a", "split": "train",
        "scenario_id": 0,
        "instruction": "당신은 친절한 한국어 교육 튜터입니다. 학습자의 질문에 정확하고 이해하기 쉽게 답하세요.",
        "map": lambda r: (r.get("instruction", ""), r.get("output", "")),
        "license": "CC-BY-NC-4.0(추정)", "rows": 21155,
        "note": "일반 한국어 instruction 베이스 (전 시나리오 공통 토대)",
    },
    "socratic": {
        "hf_id": "JosephLee/korean-socratic-qa", "split": "train",
        "scenario_id": 1,
        "instruction": "시나리오[1]: 소크라테스식 문답법으로 학생을 가이드하세요. 정답을 바로 주지 말고 힌트와 역질문으로 스스로 답을 찾게 유도합니다.",
        "map": lambda r: (r.get("input", ""), r.get("target", "")),
        "license": "확인필요", "rows": 105728,
        "note": "시나리오 1(소크라테스 문답) 직결 — input/target",
    },
    "math": {
        "hf_id": "kuotient/orca-math-word-problems-193k-korean", "split": "train",
        "scenario_id": 6,
        "instruction": "시나리오[6]: 수학 문제를 단계별로 풀이하세요. 풀이 과정을 차근차근 보여주고 최종 답을 명확히 제시합니다.",
        "map": lambda r: (r.get("question", ""), r.get("answer", "")),
        "license": "CC-BY-SA-4.0", "rows": 193789,
        "note": "시나리오 6(수학 단계별 풀이/채점 토대) — question/answer",
    },
    "empathy": {
        "hf_id": "jojo0217/korean_safe_conversation", "split": "train",
        "scenario_id": 11,
        "instruction": "시나리오[11]: 학생의 정서적 피로에 공감하고 안전하고 건설적으로 대화하세요.",
        "map": lambda r: (r.get("instruction", "") or r.get("input", ""), r.get("output", "")),
        "license": "Apache-2.0", "rows": 26979,
        "note": "시나리오 11/12/15(정서지원·공감 대화) — 라이선스 가장 자유로움",
    },
    "edu": {
        "hf_id": "neuralfoundry-coder/aihub-korean-education-instruct-sample", "split": "train",
        "scenario_id": 17,
        "instruction": "시나리오[17]: 교육 전문가로서 학생 데이터를 분석하고 교육적 조언을 제공하세요.",
        "map": lambda r: _from_conversations(r.get("conversations", [])),
        "license": "CC-BY-NC-SA-4.0", "rows": 6000,
        "note": "시나리오 2/16/17(교육 상담·분석) — AI Hub 기반 conversations 포맷",
    },
    # ── 공백 시나리오를 '검색'으로 보강한 실제 데이터셋 (합성 아님) ──────────
    "roleplay": {
        "hf_id": "huggingface-KREW/korean-role-playing",
        "config": "general-roleplay-data", "split": "train",
        "scenario_id": 9,
        "instruction": "시나리오[9]: 지정된 인물/캐릭터의 어조와 상황·가치관을 반영해 1인칭으로 답변하세요.",
        "map": lambda r: _from_roleplay(r.get("text")),
        "license": "Apache-2.0", "rows": 32367,
        "note": "시나리오 9(역사/인물 롤플레잉) — text=대화 리스트(JSON)",
    },
    "translation": {
        "hf_id": "bawin/korean-english-translation-1k", "split": "train",
        "scenario_id": 19,
        "instruction": "시나리오[19]: 한국어 문장을 자연스러운 영어로 번역하세요.",
        "map": lambda r: (r.get("korean", ""), r.get("english", "")),
        "license": "확인필요", "rows": 1000,
        "note": "시나리오 19(번역) — korean/english",
    },
    "reasoning": {
        "hf_id": "SabaPivot/KMMLU-Summarized-Chain_of_Thought",
        "config": "Agricultural-Sciences", "split": "train",
        "scenario_id": 3,
        "instruction": "시나리오[3]: 문제를 풀 때 사고 과정을 단계적으로 보여주고 정답을 제시하세요.",
        "map": lambda r: _from_kmmlu_cot(r),
        "license": "확인필요", "rows": 6962, "lang": "ko",
        "note": "시나리오 3(메타인지/단계적 추론) — KMMLU chain_of_thought (빈 CoT 자동 제외)",
    },
    # ── 영어 데이터셋으로 보강 (cross-lingual 행동 학습; 출력 한국어화는 README 참고) ──
    "debug_en": {
        "hf_id": "taisazero/socratic-debugging-benchmark", "split": "train",
        "scenario_id": 8,
        "instruction": "시나리오[8]: 정답 코드를 주지 말고 소크라테스식 질문/힌트로 디버깅을 유도하세요.",
        "map": lambda r: (r.get("lm_input", ""), r.get("lm_target", "")),
        "license": "MIT", "rows": 993, "lang": "en",
        "note": "시나리오 8(코딩 디버깅 힌트형) — lm_input/lm_target, 소크라테스 디버깅",
    },
    "science_en": {
        "hf_id": "allenai/sciq", "split": "train",
        "scenario_id": 10,
        "instruction": "시나리오[10]: 과학 개념을 근거와 함께 설명하고 정답을 제시하세요.",
        "map": lambda r: (r.get("question", ""),
                          (r.get("support", "") + f"\n\nAnswer: {r.get('correct_answer','')}").strip()),
        "license": "CC-BY-NC-3.0", "rows": 13679, "lang": "en",
        "note": "시나리오 10(과학 교육) — question/support/correct_answer",
    },
    "lesson_en": {
        "hf_id": "xriminact/brightai_edge_lesson_plan_dataset", "split": "train",
        "scenario_id": 20,
        "instruction": "시나리오[20]: 성취 기준과 주제로 도입-전개-정리 단계별 수업 지도안을 설계하세요.",
        "map": lambda r: (r.get("user", ""), r.get("output", "")),
        "license": "확인필요", "rows": 4358, "lang": "en",
        "note": "시나리오 20(수업 지도안) — user/output(JSON 지도안)",
    },
    "motivation_en": {
        "hf_id": "to-be/annomi-motivational-interviewing-therapy-conversations", "split": "train",
        "scenario_id": 14,
        "instruction": "시나리오[14]: 동기부여 면담 기법으로 변화 동기를 끌어내고 작은 실천 목표를 제안하세요.",
        "map": lambda r: _from_conversations(r.get("conversations", [])),
        "license": "OpenRAIL", "rows": 133, "lang": "en",
        "note": "시나리오 14(동기/목표) — motivational interviewing 대화 (소규모)",
    },
    # ── 교체된 시나리오(4·5·13·18)용 실제 데이터셋 ────────────────────────
    "summary": {
        "hf_id": "daekeun-ml/naver-news-summarization-ko", "split": "train",
        "scenario_id": 4,
        "instruction": "시나리오[4]: 긴 글의 핵심을 파악해 간결하고 정확하게 요약하세요.",
        "map": lambda r: (r.get("document", ""), r.get("summary", "")),
        "license": "Apache-2.0", "rows": 27400, "lang": "ko",
        "note": "시나리오 4(요약·핵심정리) — document/summary",
    },
    "reading": {
        "hf_id": "klue/klue", "config": "mrc", "split": "train",
        "scenario_id": 5,
        "instruction": "시나리오[5]: 주어진 지문을 근거로 질문에 정확히 답하세요. 지문에 없는 내용은 지어내지 않습니다.",
        "map": lambda r: _from_klue_mrc(r),
        "license": "CC-BY-SA-4.0", "rows": 23395, "lang": "ko",
        "note": "시나리오 5(독해 질의응답) — context+question/answers",
    },
    "code": {
        "hf_id": "m-a-p/CodeFeedback-Filtered-Instruction", "split": "train",
        "scenario_id": 13,
        "instruction": "시나리오[13]: 코딩 요청에 동작하는 코드와 명확한 설명을 제공하세요.",
        "map": lambda r: (r.get("query", ""), r.get("answer", "")),
        "license": "Apache-2.0", "rows": 156526, "lang": "en",
        "note": "시나리오 13(코딩 실습·피드백) — query/answer",
    },
    "writing": {
        "hf_id": "coastral/korean-writing-style-instruct", "split": "train",
        "scenario_id": 18,
        "instruction": "시나리오[18]: 요청한 문체/형식/목적에 맞춰 글을 작성하거나 다듬어 주세요.",
        "map": lambda r: _from_sharegpt(r.get("conversations", [])),
        "license": "Apache-2.0", "rows": 28978, "lang": "ko",
        "note": "시나리오 18(글쓰기 스타일 지도) — conversations(from/value)",
    },
}


def _from_conversations(convo):
    """[{role,content}...] → (user, assistant). system 은 무시(시나리오 instruction 사용)."""
    user = next((m["content"] for m in convo if m.get("role") == "user"), "")
    asst = next((m["content"] for m in convo if m.get("role") == "assistant"), "")
    return user, asst


def _from_roleplay(text):
    """korean-role-playing 의 text 필드(대화 리스트 또는 그 JSON 문자열) → (user, assistant)."""
    convo = text
    if isinstance(text, str):
        try:
            convo = json.loads(text)
        except Exception:
            return "", text  # 파싱 불가 시 완성형으로
    if isinstance(convo, list):
        return _from_conversations(convo)
    return "", ""


def _from_sharegpt(convo):
    """ShareGPT 포맷 [{from,value}...] → (human, gpt)."""
    if isinstance(convo, str):
        try:
            convo = json.loads(convo)
        except Exception:
            return "", ""
    if not isinstance(convo, list):
        return "", ""
    human = next((m.get("value", "") for m in convo if m.get("from") in ("human", "user")), "")
    gpt = next((m.get("value", "") for m in convo if m.get("from") in ("gpt", "assistant")), "")
    return human, gpt


def _from_klue_mrc(r):
    """KLUE-MRC → (지문+질문, 첫 정답). 정답 없으면 ('','')→자동 제외."""
    ans = r.get("answers") or {}
    texts = ans.get("text") if isinstance(ans, dict) else None
    answer = (texts[0] if texts else "").strip()
    if not answer:
        return "", ""
    ctx = (r.get("context") or "").strip()
    q = (r.get("question") or "").strip()
    return f"[지문]\n{ctx}\n\n[질문] {q}", answer


_KMMLU_LETTERS = {1: "A", 2: "B", 3: "C", 4: "D"}


def _from_kmmlu_cot(r):
    """KMMLU-CoT → (문제+보기, 사고과정+정답). chain_of_thought 비면 ('','')→자동 제외."""
    cot = (r.get("chain_of_thought") or "").strip()
    if not cot:
        return "", ""
    q = (r.get("question") or "").strip()
    choices = "\n".join(f"{L}) {r.get(L, '')}" for L in ("A", "B", "C", "D"))
    ans = r.get("answer")
    letter = _KMMLU_LETTERS.get(ans, str(ans))
    inp = f"{q}\n{choices}"
    outp = f"{cot}\n\n정답: {letter}"
    return inp, outp


def fetch_one(key, cfg, per_source, seed):
    from datasets import load_dataset
    print(f"\n▶ [{key}] {cfg['hf_id']} (목표 {per_source}개, license={cfg['license']})")
    out = []
    try:
        load_kw = {"split": cfg["split"], "streaming": True}
        if cfg.get("config"):
            load_kw["name"] = cfg["config"]
        ds = load_dataset(cfg["hf_id"], **load_kw)
        # streaming shuffle 은 첫 행 전에 buffer 를 모두 채우므로(=대량 선다운로드)
        # 샘플 수에 비례한 작은 버퍼만 사용해 소량 추출도 빠르게.
        buf = max(1000, min(10000, per_source * 2))
        ds = ds.shuffle(seed=seed, buffer_size=buf)
        for row in ds:
            inp, outp = cfg["map"](row)
            inp = (inp or "").strip()
            outp = (outp or "").strip()
            if not outp or len(outp) < 5:
                continue
            out.append({
                "scenario_id": cfg["scenario_id"],
                "instruction": cfg["instruction"],
                "input": inp,
                "output": outp,
                "source": cfg["hf_id"],
            })
            if len(out) >= per_source:
                break
        print(f"  ✅ {len(out)}개 수집")
    except Exception as e:
        print(f"  ⚠️  실패: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=8000, help="소스별 최대 샘플 수")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="일부 소스만")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--list", action="store_true", help="추천 목록만 출력하고 종료")
    args = ap.parse_args()

    if args.list:
        print("📚 추천 데이터셋 (HuggingFace Hub)\n")
        for k, c in SOURCES.items():
            print(f"  [{k}] {c['hf_id']}")
            print(f"      시나리오 {c['scenario_id']} | {c['rows']:,}행 | {c['license']} | lang={c.get('lang','ko')}")
            print(f"      → {c['note']}\n")
        return

    keys = args.only or list(SOURCES)
    all_rows = []
    for k in keys:
        all_rows += fetch_one(k, SOURCES[k], args.per_source, args.seed)

    random.seed(args.seed)
    random.shuffle(all_rows)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_val = int(len(all_rows) * args.val_ratio)
    val, train = all_rows[:n_val], all_rows[n_val:]

    def dump(data, name):
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  저장: {p} ({len(data)}개)")

    print(f"\n총 {len(all_rows)}개 수집 → 분할 저장")
    dump(train, "hf_train.jsonl")
    if n_val:
        dump(val, "hf_val.jsonl")

    from collections import Counter
    dist = Counter(r["source"] for r in all_rows)
    print("\n[소스별 분포]")
    for s, n in dist.most_common():
        print(f"  {s:<55} {n}")
    print("\n✅ 완료 — 학습엔 data/hf_train.jsonl 를 사용하세요 (seed_train.jsonl 은 예시 참고용)")


if __name__ == "__main__":
    main()
