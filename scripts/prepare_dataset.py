#!/usr/bin/env python3
"""
교육용 파인튜닝 데이터셋 준비 스크립트.

기능:
  1) seed_train.jsonl 포맷 검증 (instruction/input/output 필드, scenario_id)
  2) 시나리오별 분포 통계 출력
  3) train/val 분할 저장

사용 예:
  python scripts/prepare_dataset.py --input data/seed_train.jsonl --out-dir data --val-ratio 0.1

로컬에서도, Colab 셀(!python ...)에서도 동일하게 동작합니다.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = ("instruction", "input", "output")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] {path}:{ln} JSON 파싱 실패 → {e}")
    return rows


def validate(rows: list[dict]) -> None:
    errors = []
    for i, r in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if field not in r:
                errors.append(f"  row {i}: '{field}' 필드 누락")
            elif not isinstance(r[field], str):
                errors.append(f"  row {i}: '{field}' 가 문자열이 아님")
        if "output" in r and isinstance(r["output"], str) and not r["output"].strip():
            errors.append(f"  row {i}: 'output' 이 비어 있음")
    if errors:
        raise SystemExit("[검증 실패]\n" + "\n".join(errors))
    print(f"[검증 통과] {len(rows)}개 샘플, 모든 필수 필드 정상")


def print_stats(rows: list[dict]) -> None:
    dist = Counter(r.get("scenario_id", "N/A") for r in rows)
    print("\n[시나리오별 분포]")
    for sid in sorted(dist, key=lambda x: (isinstance(x, str), x)):
        bar = "█" * dist[sid]
        print(f"  시나리오 {str(sid):>3}: {dist[sid]:>3}개 {bar}")
    lens = [len(r["output"]) for r in rows]
    print(f"\n[output 길이] 평균 {sum(lens)//len(lens)}자 / 최소 {min(lens)} / 최대 {max(lens)}")


def split_and_save(rows: list[dict], out_dir: Path, val_ratio: float, seed: int) -> None:
    random.seed(seed)
    shuffled = rows[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    val, train = shuffled[:n_val], shuffled[n_val:]

    def dump(data, name):
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  저장: {p} ({len(data)}개)")

    print("\n[분할 저장]")
    dump(train, "train.jsonl")
    if n_val:
        dump(val, "val.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/seed_train.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(in_path)
    validate(rows)
    print_stats(rows)
    split_and_save(rows, out_dir, args.val_ratio, args.seed)
    print("\n✅ 완료")


if __name__ == "__main__":
    main()
