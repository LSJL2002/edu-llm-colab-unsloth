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
}


def _from_conversations(convo):
    """[{role,content}...] → (user, assistant). system 은 무시(시나리오 instruction 사용)."""
    user = next((m["content"] for m in convo if m.get("role") == "user"), "")
    asst = next((m["content"] for m in convo if m.get("role") == "assistant"), "")
    return user, asst


def fetch_one(key, cfg, per_source, seed):
    from datasets import load_dataset
    print(f"\n▶ [{key}] {cfg['hf_id']} (목표 {per_source}개, license={cfg['license']})")
    out = []
    try:
        ds = load_dataset(cfg["hf_id"], split=cfg["split"], streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10000)
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
            print(f"      시나리오 {c['scenario_id']} | {c['rows']:,}행 | {c['license']}")
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
