# 🎓 Yonsei Colab Studio

Google Colab 무료 GPU에서 **소형 LLM을 교육용으로 파인튜닝**하는 실험 환경입니다.
[Unsloth](https://github.com/unslothai/unsloth)를 사용해 무료 T4 GPU에서도 OOM 없이 빠르게 학습합니다.

> GPT-2가 아니라 더 똑똑한 최신 초경량 모델(Qwen2.5-1.5B / Llama-3.2-1B)을 기본값으로 사용합니다.

---

## 🚀 빠른 시작 (3단계)

1. **GitHub에 업로드** → `notebooks/unsloth_edu_finetune.ipynb` 를 [Google Colab](https://colab.research.google.com)에서 엽니다.
   (Colab → 파일 → 노트 업로드, 또는 GitHub 탭에서 저장소 열기)
2. 상단 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 선택
3. **런타임 → 모두 실행** (Run all)

데이터는 노트북 2번 셀에서 `git clone` 하거나 `train.jsonl`을 직접 업로드합니다.

---

## 📁 구조

```
yonsei-colab-studio/
├── README.md
├── requirements.txt
├── notebooks/
│   └── unsloth_edu_finetune.ipynb   # ⭐ 메인 Colab 파인튜닝 노트북
├── data/
│   ├── scenarios.json               # 20대 교육 시나리오 정의 (선정이유·기대치 포함)
│   ├── seed_train.jsonl             # 시나리오별 시드 학습 데이터 (직접 늘려가세요)
│   ├── train.jsonl / val.jsonl      # prepare_dataset.py 가 생성
└── scripts/
    ├── prepare_dataset.py           # 검증·통계·train/val 분할
    └── build_notebook.py            # 노트북(.ipynb) 재생성기
```

---

## 🤖 추천 모델 (가장 작은 것부터)

| 모델 | 크기 | 특징 |
|------|------|------|
| `unsloth/Qwen2.5-0.5B-Instruct` | 0.5B | 가장 빠름, 한국어 가능 |
| `unsloth/Qwen2.5-1.5B-Instruct` | 1.5B | **기본값**. 한국어 우수, 가성비 최고 |
| `unsloth/Llama-3.2-1B-Instruct` | 1B | 메타 1B급 최고 성능 |

노트북 3번 셀의 `MODEL_NAME` 한 줄만 바꾸면 됩니다.

---

## 🧪 데이터 작업 (로컬)

```bash
# 시드 데이터 검증 + 시나리오 분포 통계 + train/val 분할
python scripts/prepare_dataset.py --input data/seed_train.jsonl --out-dir data --val-ratio 0.1
```

데이터 포맷 (JSONL 한 줄):
```json
{"scenario_id": 1, "instruction": "시나리오[1]: 소크라테스식 문답법으로...", "input": "학생 질문", "output": "튜터 답변"}
```
- `instruction` → 시스템 프롬프트(시나리오 유형 명시) → 모델이 상황을 **구별**하는 핵심
- `input` → 학생/사용자 발화
- `output` → 모델이 학습할 이상적 답변

---

## 📜 20대 교육 시나리오

`data/scenarios.json` 에 전체 정의(시스템 프롬프트, 데이터 설계, 선정이유, 기대치)가 있습니다.

| # | 단계 | 시나리오 |
|---|------|----------|
| 1 | 인지능력 | 소크라테스식 문답 |
| 2 | 인지능력 | 눈높이 비유 설명 |
| 3 | 인지능력 | 메타인지 자극 (오답 피드백) |
| 4 | 인지능력 | 난이도 동적 조절 |
| 5 | 인지능력 | 개념 간 마인드맵 연결 |
| 6 | 교과특화 | 수학 서술형 단계별 채점 |
| 7 | 교과특화 | 영어 에세이 교정 |
| 8 | 교과특화 | 코딩 디버깅 (힌트형) |
| 9 | 교과특화 | 역사 인물 롤플레잉 |
| 10 | 교과특화 | 과학 실험 안전 가이드 |
| 11 | 정서지원 | 학습 슬럼프 상담 |
| 12 | 정서지원 | 과정 중심 칭찬 |
| 13 | 정서지원 | 주의 집중 게이미피케이션 |
| 14 | 정서지원 | 스몰 스텝 목표 설정 |
| 15 | 정서지원 | 느린 학습자 언어 배려 |
| 16 | 교사지원 | 시험 문제 자동 생성 |
| 17 | 교사지원 | 학생 종합의견 초안 |
| 18 | 교사지원 | 학부모 안내문 문체 변환 |
| 19 | 교사지원 | 다국어 가정통신문 |
| 20 | 교사지원 | 수업 지도안 설계 |

---

## ⚠️ 참고

- 일부 온라인 가이드의 `unsloth-studio` pip 패키지, localtunnel 비밀번호 방식 등은 부정확합니다.
  이 저장소는 **Unsloth 정식 노트북 방식**(FastLanguageModel + LoRA + SFTTrainer)을 사용합니다.
- 시드 데이터는 시나리오당 1~2개입니다. 실제 효과를 보려면 **시나리오당 30~100개**로 늘리세요.
