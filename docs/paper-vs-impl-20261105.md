# TMEM 논문 vs 우리 구현: LongMemEval-S 재현 격차 원인 점검 (2026-11-05)

- 논문: Ren et al., *Scaling Self-Evolving Agents via Parametric Memory* (TMEM), arXiv:2606.04536v1. arXiv HTML 본문과 부록(A–E) 전체를 읽음.
- 코드: `/Users/khope@sionic.ai/work/tmem-repro` (코드 수정·실행 없음. `runs/gpu/*.jsonl`만 로컬에서 분석).
- 우리 결과 (Qwen3-4B, LongMemEval-S 앞 50문항, greedy, 1회): TMEM F1 9.8 / EM 6.0, no-memory F1 8.4 / EM 0.0.
- 논문 Table 1 (RL 전, Qwen3-4B): TMEM 41.24 / 25.54, no-memory 5.30 / 0.00.

## 0. 요약

1. **no-memory는 재현됐다** (8.4/0.0 vs 5.30/0.00). 채점과 데이터 로딩의 큰 틀은 맞다. 격차는 TMEM 경로에서만 생긴다.
2. **Δ에 쓴 사실이 거의 읽히지 않는다.** 우리 run에서 gold 답이 추출 QA에 들어간 에피소드는 50개 중 20개다. 그중 맞힌 것은 2개뿐이다. F1≥0.3인 5문항 가운데 3문항은 증거가 마지막 working window(텍스트)에 남아 있던 경우다. Δ만으로 맞힌 것은 2/50이다. write마다 SGD는 약 3.7 step이고, 마지막 step loss는 1.9–2.7에 머문다. 최종 ‖Δ‖는 평균 0.19로, 사실을 외우지 못했다.
3. **논문은 답변 시 QA 텍스트를 보여 줬을 가능성이 크다.** 정황은 다음과 같다.
   - Fig. 8 캡션이 "retrieved memory QA pairs"다.
   - Fig. 11–13에서 모델 답 "$225 + $150 + $120 = $495"가 QA 쌍들을 더한 형태다.
   - Abstract와 Conclusion에 "read from context and written into fast model parameters"라고 쓰여 있다.
   - Table 5에서 supervision을 "Free-form summary"로 바꿔도 35.44가 나온다. 이는 MemAgent 36.45와 거의 같다.
   - Table 4에서 random init도 38.54다.
   - 반면 Eq. 6 직후 본문은 "no textual memory is carried forward"라고 하여 서로 모순된다. 답변 프롬프트는 논문에 없다.
4. **속도 6.55 s/ep(Table 7)는 우리 파이프라인(HF `generate`, 1 write ≈ 20.6 s)으로는 설명이 안 된다.** 논문에는 vLLM이나 batched generation 언급이 없다. 오히려 "batch size one"이라고 쓰고, peak 22.9 GB를 보고한다. vLLM 기본 설정이라면 KV pool 때문에 peak가 훨씬 컸을 것이다.

## 1. 항목별 비교표

| 항목 | 논문 (인용: 절/그림/표 번호) | 우리 구현 (파일:줄) | 같음/다름/논문에 없음 | 성능 영향 추정 |
|---|---|---|---|---|
| Base model | "Qwen3-4B and Qwen3-8B backbones" (Sec. 5.1 첫 문단). Base인지, hybrid Qwen3-4B인지, Instruct-2507인지는 논문에 없음. no-memory가 LongMemEval-S에서 78.5 GB(Table 7)이고 "the raw context grows throughout the episode"라고 설명한다. ~115k 토큰을 넣었다면 32k를 넘는 문맥(YaRN 또는 2507)이 필요하다. | `scripts/run_longmemeval.py:39,61` `Qwen/Qwen3-4B` (hybrid, native 32k) | 논문에 없음 | 중간 (2507 Instruct는 거절이 적고 장문맥이 native) |
| Table 1이 RL 전인가 | "We first evaluate each memory strategy before applying the RL phase." (Sec. 5.2). RL 후 TMEM LME는 "+4.92 F1, +4.84 EM" (Sec. 5.3, Fig. 3b) → 약 46.2 / 30.4 | RL 없음 (README 표) | 같음 (Table 1 = RL 전 θ0) | – |
| Thinking 모드 | 논문에 없음. E.2 모델 답 "$225 + $150 + $120 = $495"로 보아 짧은 계산을 노출한다. | `tmem/agent.py:74-76`, `tmem/update.py:32-34` `enable_thinking=False` | 논문에 없음 | 낮음–중간 |
| 디코딩 | 논문에 없음. "mean over 3 independent runs", std ±0.4 (Sec. 5.1 Metrics, Table 1). greedy라면 std가 거의 0이어야 하므로 sampling으로 추정 | `tmem/agent.py:78-83` greedy, 1회 | 논문에 없음 (추정상 다름) | 낮음 |
| LoRA 위치/rank | "LoRA rank to r=6 … only to the FFN projection matrices (gate_proj, up_proj, and down_proj) in the last 4 Transformer layers; attention projections are left unchanged" (Sec. 5.1 TMEM implementation) | `tmem/svd_lora.py:19,110-111,123-131` | 같음 | – |
| SVD init / A 고정 | "B0=0, A0=Σ_r V_r^T" (Eq. 7, Sec. 3.2). "A=A0 is frozen, and only the coefficient matrix B is updated online" (Sec. 5.1) | `tmem/svd_lora.py:34-41` | 같음 | – |
| LoRA scaling (alpha/r) | 논문에 없음. PEFT로 구현했다면 `lora_alpha/r` 배율이 곱해져 실효 step size가 달라진다. | `tmem/svd_lora.py:23,47` `scale=1.0` | 논문에 없음 | **높음** (실효 lr를 좌우함, 아래 진단 참고) |
| B dtype | 논문에 없음 | `tmem/svd_lora.py:39` float32 | 논문에 없음 | 낮음 |
| Optimizer / lr / epoch / batch | "online SFT with SGD, learning rate 5×10⁻⁴, 5 epochs, and batch size 16" (Sec. 5.1) | `tmem/update.py:22-24,69` | 같음 | – (값은 같지만 결과 Δ가 작음) |
| step 수 | 논문에 없음 (QA 수에 따라 달라짐). Fig. 11–12의 LME 사례는 에피소드 전체 QA가 20쌍 | 우리 run: write당 QA 6.1쌍 → epoch당 1 batch → write당 평균 3.7 step (0-QA write 포함) | 논문에 없음 | 높음 (아래 진단) |
| loss 마스킹 / reduction | "the extracted JSON QA supervision is parsed into instruction–answer pairs and used for online SFT" (Sec. 5.1). 마스킹, 템플릿, mean/sum은 논문에 없음 | `tmem/update.py:29-40` user turn=질문, assistant=답+eos, 답 토큰만 loss. `:80` HF 기본 token-mean. **system prompt 없음** | 논문에 없음 | 중간 |
| 학습/추론 포맷 일치 | 논문에 없음 | 학습: system 없는 bare chat `[user:Q]` (`tmem/update.py:31`). 추론: 긴 system prompt + 최대 12k `<history>` + `Current date` (`tmem/agent.py:170-173`, `tmem/prompts.py:44-49`) | 논문에 없음 | 중간–높음 (Δ 회상이 문맥 분포 차이로 약해질 수 있음) |
| Δ 누적 / 에피소드 리셋 | "Fast-weight updates are cumulative across triggers within an episode: each trigger starts from the current B rather than resetting to zero" (Sec. 5.1). 매 에피소드 Δ0=0 (Eq. 7, "rollout begins from the original base policy") | `tmem/agent.py:119-120` 에피소드마다 reset, 에피소드 안에서는 누적 | 같음 | – |
| 추출 프롬프트 (user, d) | Fig. 1 전문 | `tmem/prompts.py:16-30` 원문과 같음. 두 번째 줄의 비문 "Given the problem to solve, previous conversation history."까지 동일 | 같음 | – |
| 추출 system 프롬프트 | Fig. 6: question, qa_history, session을 넣음 | `tmem/prompts.py:8-14` | 같음 (태그 줄바꿈만 다름) | 낮음 |
| 추출 입력의 qa_history 길이 | Fig. 6 `{qa_history}`. 개수 제한은 논문에 없음. Fig. 14 캡션 "search-task QA history in original JSON form" | `tmem/prompts.py:52` 최근 50쌍, `tmem/agent.py:100` | 논문에 없음 | 낮음–중간 (우리 모델이 히스토리를 그대로 베껴 dup이 50 에피소드에서 1,313쌍, 에피소드당 약 26쌍) |
| QA 개수 | "Generate QA pairs adaptively" (Fig. 1). 고정 개수는 논문에 없음. LME 사례는 20쌍 (Fig. 11 "QA pairs 1-15", Fig. 12 "16-20") | 에피소드당 55쌍 유지 + 26쌍 dup 제거, write당 약 6쌍 | 논문에 없음 (사례 기준으로는 우리가 더 많음) | 낮음 |
| dedup | 논문에 없음 (프롬프트 요구사항 6 "Avoid duplicate…"만 있음) | `tmem/prompts.py:77-85` 응답 내부 dedup, `tmem/agent.py:105-107` 과거 QA와 정확히 일치하면 제거 | 논문에 없음 (우리만 추가) | 낮음 |
| 추출 max_new_tokens | 논문에 없음 (RL의 "maximum response length 1024 tokens", Sec. 5.1 RL) | `tmem/agent.py:29` 1024 | 논문에 없음 | 낮음 |
| Trigger 규칙 | "Memory operations are triggered … where ℓ(h_{t_i})+ℓ(m_{t_i})>L_max" (Sec. 3.1). "L_max=12288 for LongMemEval-S" (Sec. 5.1). Fig. 5b에서 12288이 최적 | `tmem/agent.py:161` 다음 조각을 넣으면 초과할 때 넣기 **전에** trigger → chunk ≤ L_max | 거의 같음 (경계 처리만 다름) | 낮음 |
| 청크 단위 | 논문에 없음 (대화 과제에서 h에 무엇을 어떤 단위로 append하는지 없음). Fig. 6은 "one conversation session"이라 부르지만 실제로는 L_max 창 | 세션 단위 append, L_max를 넘는 세션은 토큰 창으로 분할 (`tmem/agent.py:87-93,155-164`) | 논문에 없음 | 낮음 |
| 세션 표기 | "the model receives the full timestamped chat history" (Sec. 5.1 Conversational memory). 렌더링 형식은 논문에 없음 | `tmem/data.py:25-30` `[Session date: …]` + `role: content` | 같음 (형식은 논문에 없음) | 낮음 |
| trigger 후 텍스트 이월 | Eq. 6: TMEM은 `(∅, T(Δ,a), ∅)`. "it is absorbed into Δ through T, while no textual memory is carried forward" (Eq. 6 직후). 그러나 Fig. 6 추출 프롬프트는 qa_history 텍스트를 계속 넣는다 (이미 텍스트 이월) | `tmem/agent.py:153` h 비움, memory="" 유지. QA는 추출 프롬프트에만 텍스트로 들어감 | 같음 (논문 본문 기준) | – |
| **답변 시 문맥** | 최종 응답은 "(q,h_T,m_T)"에서 (Sec. 3.1 끝). **답변 프롬프트는 논문에 없음.** "answer templates, and scoring rules are shared" (Sec. 5.1 Baselines). 반면 Fig. 8 캡션은 "A LoCoMo case with **retrieved memory QA pairs** and the final question". Fig. 13 QA에는 "when multiple QA pairs contain separate amounts … then sum them"이 있다. Abstract: "not only compresses history into explicit memory but also absorbs … into fast LoRA weights". Sec. 7: "both be read from context and written into fast model parameters" | `tmem/prompts.py:44-49` + `tmem/agent.py:166-173`: 마지막 미-trigger h_T(≤12k)와 질문만. **QA 텍스트도 retrieval도 없음** | **다름 가능성 큼** (논문 내부 모순) | **매우 높음** |
| 답변 길이 / 스타일 | 논문에 없음. 검색 과제 프롬프트는 "concise, contain only essential words, and avoid any explanations" (Fig. 7) | `tmem/prompts.py:45-46` 비슷한 지시, `tmem/agent.py:31` 64 tokens. 실제 예측: TMEM 평균 6.8단어, none 38.7단어 (gold 2.4단어). 거절형 12/50 | 논문에 없음 | 중간 (F1 precision 손실 + 거절) |
| no-memory 문맥 | "keeps only the current working context" (Sec. 5.1 Baselines). 78.5 GB, 8.73 s/ep (Table 7) → 거의 전체 haystack | `tmem/agent.py:168-169`, `scripts/run_longmemeval.py:45` 마지막 32k 토큰 | 다름 | 낮음 (결과가 비슷함) |
| 지표 정규화 | "token-level F1 and Exact Match (EM) after lowercasing, stripping punctuation, and normalizing whitespace. Scores are computed per instance and then averaged" (Sec. 5.1 Metrics) | `tmem/metrics.py:11-32` 같음 (관사 제거 없음, 논문도 언급 없음) | 같음 | 낮음 |
| 평가 split / 표본 | "the released 500-instance evaluation set under the full-haystack protocol" (Sec. 5.1 Conversational memory). 3회 평균 (Sec. 5.1 Metrics). cleaned 판본 여부는 논문에 없음 | `tmem/data.py:10-12` `longmemeval_s_cleaned.json`. `--limit 50` = 파일 앞 50개 = **전부 single-session-user** (500개 중 70개 유형). abstention(`_abs`) 0개 | 다름 | 중간 (비교 불가 수준의 표본 편향. 방향은 불명) |
| 속도 / 엔진 | "Each row uses Qwen3-4B under the same local evaluation pipeline and batch size one. Wall-clock time … includes memory operations … as well as SVD adapter initialization on the first episode; GPU memory is the peak allocated memory" (Sec. 5.5, Table 7). TMEM 6.55 s / 22.9 GB, MemAgent 5.12 s / 12.7 GB. vLLM, batched generation, KV 재사용 언급은 **논문에 없음** | HF `generate`, write당 20.6 s × 9.1 write ≈ 189 s/ep | 논문에 없음 | 정확도 영향 낮음, 속도 영향 높음 |

## 2. 진단 근거 (`runs/gpu/lme_tmem_4b_50.jsonl` 로컬 분석)

| 지표 | 값 | 해석 |
|---|---|---|
| gold가 어떤 추출 QA의 답에 포함 (substring 또는 F1≥0.5) | 20/50 | 추출 recall 40%. 추출 단계 손실이 30/50 |
| 그 20개 중 정답 (F1≥0.5) | 2 | **흡수(T) 또는 회상 단계에서 18/20 손실** → 주 병목 |
| F1≥0.3인 5문항 중 증거가 마지막 h_T(텍스트)에 있던 경우 | 3 | Δ로만 맞힌 것은 2/50 |
| write별 첫/마지막 step loss | write0 7.29→2.73, write8 2.90→1.88 | 짧은 답을 greedy로 회상하려면 loss가 1보다 훨씬 낮아야 함. 덜 학습됨 |
| write당 SGD step | 평균 3.66 | QA 약 6쌍 < batch 16 → epoch당 1 step |
| 최종 ‖Δ‖ (B Frobenius) | 0.19 | 매우 작음 |
| 거절형 답변 ("I don't have access…", "None of the…") | 12/50 (TMEM) | base 모델의 거절 성향이 Δ로 덮이지 않음 |

참고: 논문 Table 5의 "Raw next-token prediction" LME F1 10.37은 우리 TMEM 9.8과 거의 같다. 즉 우리 Δ는 "효과가 거의 없는 쓰기" 수준이다.

## 3. 시도할 수정 Top 5 (기대 효과 순)

1. **답변 프롬프트에 누적 QA를 텍스트로 넣는 변형을 추가한다** (`tmem/agent.py:170-171`, `tmem/prompts.py:44-49`).
   - 두 변형을 비교한다: `qa_log` 전체를 넣는 것과, 질문으로 retrieval한 top-k를 넣는 것. 각각 Δ 있음/없음 2×2로 돌린다.
   - 근거: Fig. 8 캡션 "retrieved memory QA pairs", Fig. 11–13의 답 형태, Abstract/Sec. 7 문구, Table 5 summary-supervision 35.44.
   - 기대: 추출 recall 40%만으로도 F1 30–40대가 가능하다. 논문 수치와 맞으면 "논문 TMEM = QA 텍스트 + Δ"로 문서화한다. 순수 parametric 수치는 별도로 보고한다.
2. **흡수 강도를 높이고 측정한다** (`tmem/update.py`, `tmem/svd_lora.py:23`).
   - 먼저 write 직후 학습한 질문 그대로 greedy 회상률(in-episode recall)을 로깅한다.
   - 그 회상률이 높아질 때까지 실효 step을 키운다: `scale`(alpha/r 상당), lr, epoch 수, 또는 batch < QA 수로 step 수를 늘림. loss는 sum 또는 per-example 평균.
   - 진행 중인 lr 1e-3 / 10 epoch / last 8 layer 런의 첫 3문항 F1 19.0도 이 방향을 지지한다. 단 범용 능력 저하는 같이 측정해야 한다.
3. **학습/추론 포맷을 맞춘다** (`tmem/update.py:29-40`).
   - SFT 예시에도 답변과 같은 system prompt("Answer the question about the user's past conversations…")를 넣는다.
   - 질문 paraphrase 1–2개로 증강해 질문 표현 차이에 견고하게 만든다.
   - 거절 12/50도 이 경로로 줄어들 가능성이 있다.
4. **평가 프로토콜을 논문과 맞춘다** (`scripts/run_longmemeval.py:74`, `tmem/data.py:40-62`).
   - 앞 50개(전부 single-session-user) 대신 6개 유형 층화 표본 또는 500개 전체를 쓴다. abstention도 포함한다.
   - 가능하면 sampling 3회 평균을 낸다. 모델 체크포인트(Qwen3-4B hybrid vs Qwen3-4B-Instruct-2507)를 바꿔 no-memory 5.30에 더 가까운 쪽을 확인한다.
   - 답을 짧게 강제한다(현재 TMEM 6.8단어, none 38.7단어).
5. **추출 품질과 속도를 개선한다** (`tmem/agent.py:96-110`, `tmem/prompts.py:52`).
   - qa_history를 질문만 넣거나 개수를 줄여 복사형 dup(에피소드당 26쌍)을 줄인다. 그만큼 생성 토큰과 시간이 준다.
   - 추출 recall 40% 개선은 1–3번 이후에 의미가 커진다.
   - 속도: 추출 생성을 vLLM으로 옮긴다. B-only SVD-LoRA는 A=A0, B를 표준 LoRA 어댑터로 내보낼 수 있으므로 write마다 어댑터를 교체하면 된다. 논문의 6.55 s/ep는 HF 순차 생성으로는 도달하기 어렵다. 다만 논문은 엔진을 밝히지 않았다.

## 4. 논문 자체의 불명확·모순 지점 (저자 문의 후보)

- Eq. 6 "no textual memory is carried forward"는 Fig. 6이 qa_history를 추출 프롬프트에 넣는 것과 충돌하고, Fig. 8 "retrieved memory QA pairs"와도 충돌한다.
- 답변 프롬프트, Qwen3 체크포인트, thinking, 디코딩, LoRA scaling, loss reduction이 모두 논문에 없다.
- Table 7의 6.55 s/ep와 22.9 GB 조합이 어떤 엔진으로 나왔는지 논문에 없다.
