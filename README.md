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

## 📥 학습 데이터 — HuggingFace Hub에서 (권장)

손으로 만든 예시(`seed_train.jsonl`)는 **시나리오별 목표 톤 참고용**일 뿐, 실제 파인튜닝에는
HuggingFace Hub의 **실제 한국어 교육 데이터셋**을 받아 씁니다.

```bash
pip install datasets
# 소스별 8000개씩 streaming 추출 → data/hf_train.jsonl 생성
python scripts/fetch_hf_datasets.py --per-source 8000 --val-ratio 0.05
python scripts/fetch_hf_datasets.py --list            # 추천 목록만 보기
python scripts/fetch_hf_datasets.py --only socratic math   # 일부 소스만
```

### 추천 데이터셋 (시나리오 매핑)

| 소스 키 | HF 데이터셋 | 행수 | 라이선스 | 매핑 시나리오 |
|---------|-------------|------|----------|---------------|
| `general` | [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) | 21k | CC-BY-NC-4.0(추정) | 0 · 일반 instruction 베이스 |
| `socratic` | [JosephLee/korean-socratic-qa](https://huggingface.co/datasets/JosephLee/korean-socratic-qa) | **105k** | 확인필요 | 1 · 소크라테스 문답 |
| `math` | [kuotient/orca-math-word-problems-193k-korean](https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean) | **193k** | CC-BY-SA-4.0 | 6 · 수학 단계 풀이 |
| `empathy` | [jojo0217/korean_safe_conversation](https://huggingface.co/datasets/jojo0217/korean_safe_conversation) | 27k | **Apache-2.0** | 11·12·15 · 정서지원 |
| `edu` | [neuralfoundry-coder/aihub-korean-education-instruct-sample](https://huggingface.co/datasets/neuralfoundry-coder/aihub-korean-education-instruct-sample) | 6k | CC-BY-NC-SA-4.0 | 2·16·17 · 교육 상담·분석 |

> ⚠️ **라이선스**: 상업적 사용 시 `empathy`(Apache-2.0)가 가장 자유롭습니다. KoAlpaca/AI Hub 계열은
> 비상업(NC) 조건이 있을 수 있으니 배포 전 각 데이터셋 카드를 확인하세요. 연구·교육 목적 파인튜닝엔 무방합니다.

### 통합 데이터 포맷 (JSONL 한 줄)
```json
{"scenario_id": 1, "instruction": "시나리오[1]: 소크라테스식 문답법으로...", "input": "학생 질문", "output": "튜터 답변", "source": "JosephLee/korean-socratic-qa"}
```
- `instruction` → 시스템 프롬프트(시나리오 유형 명시) → 모델이 상황을 **구별**하는 핵심
- `input` → 학생/사용자 발화 · `output` → 학습할 이상적 답변

### 아직 실제 데이터가 부족한 시나리오 (3·4·5·8·9·10·13·14·18·19·20)
HF에 직결 데이터가 적은 시나리오입니다. `seed_train.jsonl`의 예시를 **few-shot 프롬프트**로 활용하거나,
추후 합성 데이터 생성(LLM)으로 보강하세요. (다음 단계 후보)

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
