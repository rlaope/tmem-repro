# 공유 LoRA fast weight에서 기억 하나를 지우기: 관련 연구 (ARS lit-review, 2026-11-05)

**대상 설정.** TMEM식 fast weight (Ren et al., 2026, arXiv:2606.04536)를 쓴다.
- A는 SVD로 초기화한 뒤 고정하고, B만 SGD로 학습한다.
- write 하나는 가상 인물 한 명에 대한 QA 쌍으로 SGD를 몇 epoch 돌리는 것이고, 이전 QA를 rehearsal로 섞는다.
- write k는 기록해 둔 update ΔB_k를 α배 해서 빼는 방식으로 지운다. 비교 기준은 k 없이 재학습한 oracle이다.

**방법.**
- academic-research-skills `deep-research` 스킬을 lit-review 모드로, 질문 없이 진행했다.
- 문헌 고찰 형식은 narrative/integrative review로 내가 정했다. 사용자 응답을 받을 수 없어 review-form note는 건너뛰었다.
- 검색은 4개 lane에서 병렬로 했다: (A) task arithmetic과 merging, (B) LLM unlearning과 평가, (C) 저장된 update로 하는 삭제·sharding·선형화, (D) model editing과 지식 저장.
- 모든 arXiv ID는 ARS `scripts/arxiv_client.py`로 arXiv API를 조회해 확인했다. 기준은 ID가 resolve되고 제목 유사도가 0.70 이상인 것이다.
- 확인하지 못한 항목은 참고문헌에서 빼고 §6에 따로 적었다.
- 이전 노트(`~/.hermes/profiles/kami/workspace/research/pmem-forget/related-work-20261030.md`)에 있는 항목은 반복하지 않는다. 해당 노트에 있는 것: CLS-Ledger, Bemula, AUBench, K-Bench, RPMem, "Where Should Deletion Live?", MemScope 등.

---

## 요약 (5줄)

1. **저장한 update 빼기는 새로운 방법이 아니다.** Amnesiac ML(Graves et al., 2020)이 이미 batch별 update를 기록해 두고 빼는 방법을 제안했다. 이 방법이 재학습과 같지 않다는 점(path dependence)도 그 논문 자체와 Thudi et al.(2021)의 verification error 분석, FedEraser 등이 정식으로 다뤘다. 문헌에서 "정확한 삭제"는 격리(SISA, AdapterSwap, O-LoRA, GRACE, MELO)나 선형 memory(Larimar의 α=−1 downdate)로만 얻는다.
2. **"공유 성분" 효과 자체는 알려져 있다.** 다만 고정된 이름이 없다. Ethos는 "general knowledge", Ext-Sub는 공유 skill, Twin-Merging은 shared/exclusive knowledge, Iso-C는 common subspace, CART는 mean task vector, DeVec은 shared/unique component, LATA는 instruction-following component, Spurious Forgetting은 task alignment라고 부른다. 이론 결과도 있다(Li et al., ICLR 2025): negation은 다른 task가 무관하거나 모순될 때만 안전하다.
3. **"관련 없는 write의 update를 빼는" decoy 대조군은 찾지 못했다.** Ilharco et al.(2023)의 random vector 대조군은 우리 random-direction 결과와 같다. decoy 대조군은 우리 쪽 새 진단으로 보인다.
4. **삭제 뒤 남의 값을 답하는 것(borrowed memory)은 이상 현상이 아니다.** k를 본 적 없는 oracle도 그럴듯한 값을 지어낼 것으로 예상된다(TOFU의 perturbed answer, Yuan et al.의 untargeted oracle). 메커니즘 근거도 있다: Kang et al.(낯선 질문에 대한 기본 응답), Ghosal et al.(entity를 무시한 generic 응답), Pletenev et al.(LoRA 지식 주입 뒤 몇몇 답으로 쏠림). 한편 honesty 기준으로 보면 실패다(Gu et al., 2026, "Unlearners Can Lie").
5. **α\* < 1에는 negation λ 튜닝이라는 선례가 있다.** Ilharco는 λ를 0~1에서 탐색했고, Zhang et al.은 (IA)³에 λ=0.6을 썼고, LATA의 baseline은 0.8이다. 그러나 retrain oracle에 대해 SGD update 하나의 scale을 맞춘 연구는 찾지 못했다. 평가는 oracle 대비 분포 수준(Truth Ratio/KS, FADE), paraphrase, leak@k, relearn time(oracle 대비), proximity로 등급을 나눈 keep-set으로 하는 것이 표준이다.

---

## 1. 질문 1: LoRA·adapter memory에서 정확하고 싸게 지우기에 관한 선행 연구

### 1.1 저장한 update 빼기와 trajectory 재계산 (우리 방법과 가장 가까움)
- **Amnesiac ML** (Graves et al., 2020)
  - 민감한 데이터가 들어간 batch의 Δθ를 저장해 두고, 지울 때 θ − ΣΔθ로 뺀다. scale은 1이다.
  - 논문 스스로 한계를 적었다. b1을 되돌린 모델은 b0, b2로만 학습한 모델과 같지 않다. b2의 update가 b1이 있는 상태에서 계산됐기 때문이다.
  - 지운 batch가 적으면 차이가 작다. 지운 직후에는 남은 데이터 정확도가 잠깐 떨어지고, 소량 fine-tuning으로 회복한다.
- **Unrolling SGD** (Thudi et al., 2021)
  - verification error ‖w_unlearn − w_retrain‖를 정의했다.
  - SGD를 Taylor 전개해서 오차를 두 부분으로 나눈다. w₀에서 계산한 gradient 합은 순서와 무관해 되돌릴 수 있다. Hessian으로 얽힌 항은 되돌릴 수 없다.
  - 남는 오차는 ‖w_t − w₀‖, 곡률, 이후 step 수에 비례한다. 선형(σ=0)이면 0이다.
  - 같은 저자들의 forgeability 결과(Thudi et al., 2022)가 있으므로, weight 거리만으로 삭제를 증명할 수는 없다.
- **DeltaGrad** (Wu et al., 2020): 저장한 gradient와 L-BFGS로 leave-k-out trajectory를 보정한다. 보장은 convex 조건에서만 성립한다.
- **연합학습 쪽 결과**
  - FedEraser(Liu et al., 2021)는 남은 client update에도 대상의 영향이 섞여 있어서, 몇 라운드 calibration을 거친다.
  - Wu et al.(2022)은 대상 client의 누적 update를 뺀 뒤 distillation으로 복구한다. 빼기만으로는 남은 지식이 손상된다는 뜻이다.
- **Rewind-to-Delete** (Mu & Klabjan, 2024): checkpoint로 되돌린 뒤 남은 데이터로 학습한다. non-convex에서 certified 보장을 준다. 우리 설정으로 옮기면 "B_{k−1}로 되돌리고 write k+1..n을 replay"하는 정확한 기준선이 된다.
- **Online Learning and Unlearning** (Hu et al., 2025): online GD 중간에 삭제 요청이 들어오는 설정의 convex 이론이다.

### 1.2 Task vector와 LoRA negation
- **Task arithmetic** (Ilharco et al., 2023)
  - −λτ로 능력을 지운다. λ는 0~1에서 탐색해 control task 성능이 95% 이상 유지되는 가장 큰 값을 고른다.
  - 층마다 크기를 맞춘 random vector는 효과가 없다.
- **LoRA negation** (Zhang et al., 2023)
  - LoRA는 B의 부호만 뒤집는다(A 유지). 우리 B-only 빼기와 같은 구조다.
  - detox에서 고른 λ는 LoRA 1.0, (IA)³ 0.6이다. 부록에서 직접 확인했다.
- **F-Learning** (Ni et al., 2023): 옛 지식의 LoRA delta를 뺀 다음 새 지식을 학습한다.
- **Ext-Sub** (Hu et al., 2024): anti-expert LoRA를 통째로 빼면 fluency 같은 공유 skill도 사라진다. 그래서 expert와 공유하지 않는 부분만 뺀다.
- **MUSE** (Shi et al., 2024): Task Vector baseline은 잘 잊지 못하고 utility는 떨어진다.
- **PerTA** (Cai et al., 2026): parameter별로 task vector의 scale을 조절해 과잉 망각을 줄인다.
- **NegMerge** (Kim et al., 2025): negation은 hyperparameter에 민감하다. sign이 일치하는 원소만 뺀다.
- **MAAT** (Yagnik et al., 2026): LoRA negation에 SVD rank pruning과 retain repair를 더한다.

### 1.3 Sharding과 격리를 통한 정확한 삭제
- **SISA** (Bourtoule et al., 2021): shard, slice, checkpoint 구조이고, 해당 shard만 재학습하면 정확하다.
- **adapter를 단위로 하는 삭제**
  - AdapterSwap(Fleshman et al., 2024)과 APA(Hu et al., 2024)는 shard마다 adapter를 둔다.
  - SIFT-Masks(Kuo et al., 2025)는 merge에서 모델 하나를 빼서 정확하게 삭제한다. 다만 task 간 정보 공유로 utility를 올리는 merge일수록 정확한 삭제가 비싸진다.
  - Leave-one-out ensemble(Liu & Kalinli, 2023)도 같은 계열이다.
- **편집 단위로 격리하는 방법**: GRACE의 codebook, MELO의 편집별 LoRA, T-Patcher(Huang et al., 2023)의 neuron 추가, WISE의 side memory와 shard.
  - 모두 편집 하나를 구조적으로 떼어 낼 수 있다. 공유 LoRA와 반대되는 설계다.

### 1.4 직교·부분공간 update
- **O-LoRA** (Wang et al., 2023): task별 LoRA를 이전 LoRA와 직교하게 둔다.
- **InfLoRA** (Liang & Li, 2024): A를 설계해 고정하고 B만 학습한다. 우리 frozen-A 설정과 가장 가깝다.
- **O³** (Gao et al., 2025): 삭제 요청마다 직교 LoRA를 하나씩 쓴다.
- **SVD 직교 LoRA** (Rahulamathavan et al., 2026): LoRA를 순차로 그냥 합치면 parameter collision이 생긴다. 30개 task 뒤 정확도가 12.7%까지 떨어지고, 직교 제약을 두면 58.1%를 유지한다.
- **BID-LoRA** (Rachapudi et al., 2026)
- **LoRA−** (Liu & Chang, 2025): 이전 task의 LoRA를 빼서 drift에 강한 공간을 만든다. vision CL이다.
- **선형 메모리에서는 빼기가 정확하다**: Larimar(Das et al., 2024)는 α=+1이면 쓰기, α=−1이면 잊기다. 선형 least-squares memory여서 downdate가 정확한 해다.
- **선형화(NTK)**
  - Golatkar et al.(2020a, 2020b, 2021)과 Fast-NTK(Li et al., 2023)는 선형화된 영역에서 forgetting을 이차 문제로 풀어 거의 정확하게 만든다.
  - Tangent-space task arithmetic(Ortiz-Jimenez et al., 2023)은 weight disentanglement를 키운다.

### 1.5 Model editing을 되돌리기
- **정확한 경우는 하나뿐이다**: ΔW를 저장해 둔 ROME 단일 편집. 이 경우 빼면 비트 단위로 복원된다.
- **나머지는 정확하지 않다**
  - MEMIT은 여러 편집을 하나의 least-squares로 함께 푼다.
  - 순차 편집에서는 이후 편집이 k가 있는 상태에서 맞춰졌다.
  - Knowledge in Superposition(Hu et al., 2024)은 key들이 직교하지 않으면 간섭 항이 남는다는 것을 보였다.
  - 저장한 update 없이 되돌리는 blind reversal은 최대 94%에 그친다(Youssef et al., 2025).
  - 이후 fine-tuning이 편집을 지운다(Cheng et al., 2025).
- **편집으로 지운 정보는 남아 있다**: Patil et al.(2023)에서 ROME으로 "삭제"한 뒤에도 hidden state 공격과 paraphrase 공격으로 38%가 복원된다.
- **Editing as Unlearning** (Li et al., 2025): 거부 응답을 목표로 편집하는 WISE와 AlphaEdit이 강한 unlearning baseline이다.
- **AlphaEdit** (Fang et al., 2024): null-space 투영으로 보존할 지식을 건드리지 않는다. keep 손상을 막는 데 쓸 수 있는 후보다.

### 1.6 LLM unlearning 방법
- **Gradient ascent** (Jang et al., 2022)
- **NPO** (Zhang et al., 2024), **SimNPO** (Fan et al., 2024)
- **RMU** (Li et al., 2024, WMDP)
- **IDK 목표**: TOFU의 IDK baseline, IDK+AP(Yuan et al., 2024)
- **대체 답을 목표로 하는 방법**: WHP의 generic replacement label(Eldan & Russinovich, 2023), AltPO(Mekala et al., 2024), ROCR(Ye et al., 2025)
- **편집 계열**: ELM(Gandikota et al., 2024)

### 1.7 억제인가 삭제인가: relearning 공격
- **적은 데이터로 되살아난다**
  - benign relearning(Hu et al., 2024)
  - 관련 없는 예제 10개로 fine-tuning(Łucki et al., 2024)
  - 4-bit 양자화로 21%였던 것이 83%까지 복원(Zhang et al., 2024)
- **Deeb & Roger (2024)**
  - 지운 사실 일부로 fine-tuning하면 서로 독립인 다른 사실까지 88% 회복된다.
  - fine-tuning으로 넣은 지식에서 평가하면 robustness가 과대평가될 수 있다고 경고한다. 우리 설정이 바로 이 경우다.
- **UNDO** (Lee et al., 2025): oracle을 흉내 낸 모델도 빨리 다시 배운다. relearn 속도는 oracle과 비교해야 의미가 있다.
- **SAM** (Fan et al., 2025)
- **Unlearning Isn't Deletion** (Xu et al., 2025)
- **Unlearning vs. Obfuscation** (Sun et al., 2025): 틀린 답이나 대체 답을 주입하는 것은 제거가 아니라 지식을 추가하는 것이다.
- **Cooper et al. (2024)**와 **Yoon et al. (2026, position)**: retrain과 동등할 때만 unlearning이라 부르고, 나머지는 suppression이라 부르자고 제안한다.

---

## 2. 질문 2: 공유 성분 효과는 보고되었는가, 어떤 이름으로

**보고되었다.** 단, 우리 설정(한 사람에 대한 SGD write, 공유 LoRA, 하나씩 삭제)에서는 아니고, task vector negation과 merging 쪽에서다. 이름은 논문마다 다르다.

| 이름 | 논문 | 공유 성분을 정의하는 방법 | 삭제에 어떻게 쓰나 |
|---|---|---|---|
| "general knowledge" vs "undesired knowledge" | Ethos (Gao et al., 2024) | pretrained W의 SVD 기저에 task vector를 투영해 분류 | undesired 성분만 negation. 전체를 negation하면 일반 능력이 손상된다고 보고 |
| 공유 skill (fluency 등) | Ext-Sub (Hu et al., 2024) | anti-expert와 expert LoRA가 공유하는 부분 | 공유하지 않는 부분만 뺀다 |
| shared vs unique component | DeVec (Damirchi et al., 2025) | 여러 task vector에 걸쳐 변하지 않는 부분공간 | unique만 negation. toxicity −47%, 일반 지식 유지 |
| instruction-following component | LATA (Chen et al., 2025) | 층별로 instruction vector(instruct − base)와의 cosine | 유사도가 높은 층의 가중치를 낮춘다 |
| shared / exclusive knowledge | Twin-Merging (Lu et al., 2024) | task vector 전체의 merge를 shared로 둔다 | merge |
| common subspace | Iso-C/CTS (Marczak et al., 2025) | Σ_t Δ_t의 주요 특이벡터 | merge |
| mean task vector (centering) | CART (Choi et al., 2024) | task vector들의 평균 | 평균을 빼고 low-rank로 만든다 |
| redundant vector δ | AWD (Xiong et al., 2024) | 모든 task vector에서 빼는 공유 δ를 학습 | 직교화 |
| shared knowledge (forget과 retain이 공유) | DUSK (Jeung et al., 2025) | 데이터 수준의 중복 | 대부분의 unlearning이 공유 지식까지 손상 |
| task alignment vs knowledge | Spurious Forgetting (Zheng et al., 2025) | 성능 하락이 지식 손실이 아니라 정렬 손실에서 온다 | 하위 층 고정 |
| 간섭 일반 | TIES(sign 충돌, 작은 redundant update), TALL/Consensus("selfish", "catastrophic" weight), TSV(특이벡터 간섭), weight disentanglement가 없음(Ortiz-Jimenez et al., 2023) | — | merge |

- **이론**: Li et al.(ICLR 2025)은 비선형 transformer에서 task vector negation이 안전하려면 나머지 task가 "irrelevant or contradictory"여야 한다는 것을 보였다. 정렬된(aligned) task는 손상된다. 우리 write들은 모두 같은 QA 형식이므로 정렬된 경우다.
- **"형식은 빨리, 사실은 느리게" 배운다는 근거**
  - LIMA의 superficial alignment
  - Kung & Peng(2023): 지시문을 바꿔도 이득이 비슷하다. 즉 형식을 배운다.
  - Chat Vector(Huang et al., 2024): 형식·지시 수행 능력을 weight 방향으로 옮길 수 있다.
  - Gekhman et al.(2024): 새 지식은 늦게 학습된다.
  - Allen-Zhu & Li(2023): 증강 없이는 사실이 entity에 묶이지 않는다.
- **추천 용어**: 논문에서는 "write update의 공유(공통) 성분 = task/format alignment 성분"으로 쓰는 것을 권한다. 인용은 이렇게 묶는다.
  - 평균·공통 부분공간: Twin-Merging, Iso-C, CART
  - 전체를 negation하면 공유 성분이 손상된다: Ethos, Ext-Sub, DeVec
  - 형식/정렬로 해석: LATA, Spurious Forgetting
- **decoy 대조군**
  - 찾지 못했다. 같은 상태에서 학습한 무관한 update를 빼서 keep 손상이 target을 뺄 때와 비슷하다는 것을 보인 연구는 없었다.
  - Ilharco et al.는 random vector 대조군만 썼고, 효과가 없었다. 우리 결과와 같다.
  - Ethos, DeVec, LATA는 공유 성분을 투영이나 유사도로 추정할 뿐, 다른 task vector를 빼는 대조 실험은 하지 않았다.
  - 따라서 decoy 대조군은 새로운 진단이라고 주장할 수 있다. 주의: 이번 검색은 체계적 검색이 아니다.

---

## 3. 질문 3: 평가 관행

1. **Oracle(retrain) 비교가 기준이다.**
   - TOFU: retain model 대비 Truth Ratio 분포를 KS test로 비교해 forget quality를 낸다.
   - MUSE: privacy leakage를 retrain 기준으로 잰다.
   - FADE(Cho et al., 2025): 분포 동등성을 본다. TOFU 표준 지표에서 거의 최적처럼 보이던 방법이 FADE에서는 oracle에서 오히려 멀어지기도 한다.
2. **답 수준의 일치는 약한 증거다.**
   - TOFU의 "illusion of forgetting": greedy 답이 틀려도 정답 확률은 높게 남을 수 있다.
   - leak@k(Reisizadeh et al., 2025): 여러 번 샘플링하면 다시 나온다.
   - 횡설수설하는 모델도 KS test를 통과할 수 있다.
   - 그래서 다음을 함께 본다: 정답 대 paraphrase 대 perturbed answer의 확률, 양자화, activation probe(Patil et al., 2023; Lynch et al., 2024).
3. **Paraphrase와 형식 변형**
   - TOFU의 paraphrased answer, ORT(Ye et al., 2025, form-dependent bias), Patil의 rephrasing 공격, ROME/MEMIT의 generalization 지표.
   - Allen-Zhu & Li를 보면, paraphrase 없이 학습한 사실은 원래 질문에서만 꺼낼 수 있다. 따라서 paraphrase probe는 삭제 평가와 학습 평가 양쪽에 다 필요하다.
4. **Relearning 속도**
   - Golatkar et al.(2020a)의 re-learn time이 LLM 쪽으로 확장됐다: Hu et al.(benign), Deeb & Roger(held-out 사실), Łucki et al.(무관한 예제 10개).
   - 반드시 oracle이 같은 사실을 처음 배우는 속도와 비교해야 한다(Deeb & Roger; UNDO).
5. **Keep-set 손상은 proximity로 등급을 나눠 잰다.**
   - TOFU: retain set → real authors → world facts. TOFU의 entanglement 결과대로 가까울수록 손상이 크다.
   - RWKU(Jin et al., 2024): neighbor perturbation.
   - Knowledge holes(Ko et al., 2025): 벤치마크가 놓치는 부수 손상.
   - Thaker et al.(2024): 벤치마크를 조금만 바꿔도 손상이 더 드러난다.
6. **순차·온라인 요청**: MUSE의 sustainability 기준, Jang et al.의 sequential unlearning, continual unlearning(O³).

---

## 4. 우리 결과와의 관계

### 4.1 이미 알려진 것
- **빼기 ≠ retrain, 그리고 path dependence**: Amnesiac ML과 Thudi et al.에서 이미 다뤘다. 우리 결과를 "정확한 삭제"라고 부르면 안 된다. "근사 삭제: 저장 update 빼기"로 불러야 한다.
- **B만 negation**: LoRA 구조상 당연하다(Zhang et al., 2023).
- **random 같은 norm 방향은 효과 없음**: Ilharco et al.의 random vector 대조군과 같다.
- **λ/α를 1보다 작게 튜닝하는 관행**: Ilharco, Zhang((IA)³ 0.6), LATA baseline 0.8. NegMerge는 scale에 민감하다고 보고했다.
- **통째로 빼면 공유 성분이 같이 손상된다**: Ethos, Ext-Sub, DeVec, LATA, DUSK. 이론은 Li et al. 2025.
- **삭제 뒤 대체 값으로 답하는 것**
  - 설계 목표로 삼은 방법: WHP, AltPO, ROCR
  - oracle도 지어낸다는 지적: TOFU, Yuan et al.
  - 메커니즘: Kang et al., Ghosal et al., Pletenev et al., knowledge overshadowing(Zhang et al., 2025)
- **IDK는 별도로 설계해야 나오는 목표다**: TOFU의 IDK baseline, Editing as Unlearning, Unlearners Can Lie.
- **격리하면 정확한 삭제가 가능하다**: SISA, AdapterSwap, O-LoRA, GRACE, MELO, Larimar. 이전 노트의 CLS-Ledger도 같은 결론이다.

### 4.2 새로운 것 (현재 검색 범위 기준)
1. **설정이 새롭다.** 에이전트가 온라인으로 누적하는 공유 fast-weight memory(TMEM)에서 write 하나를 지우고 oracle과 비교한 연구는 없었다.
   - TMEM 원 논문은 삭제를 다루지 않는다.
   - 이전 노트의 RPMem, "Where Should Deletion Live?"는 영역이 다르거나 weight를 다루지 않는다.
2. **decoy-subtraction 대조군.** 같은 상태에서 학습한 무관한 write를 빼서, keep 손상이 k 고유 내용이 아니라 공유 성분 때문이라는 것을 행동 수준에서 직접 보이는 실험이다.
3. **SGD update 하나에 대한 α\***를 retrain oracle 기준으로 측정한 것. 선행 연구는 task vector 전체의 λ를 utility 제약 아래에서만 골랐다.
4. **borrowed memory를 fast-weight 삭제의 결과로 보고한 것.** 특히 빌려 온 값이 누구의 값인지(가장 가까운 사람, 가장 최근 write, 가장 많이 학습한 사람)를 추적한 연구는 없었다.

### 4.3 확인이 필요한 것 (다음 실험 후보)
- **(a) Oracle도 남의 값을 빌리는가?**
  - 빌린다면 borrowing은 oracle에 충실한 삭제의 증거다.
  - 빌리지 않는다면(예: oracle은 다른 분포를 낸다면) 우리 삭제가 남긴 흔적이나 overshadowing이다.
  - 빌려 온 값이 누구 것인지, 그 사람이 최근에 썼는지·rehearsal 빈도·속성 유사도를 함께 기록한다.
- **(b) 공유 성분 투영 실험.**
  - 모든 write의 ΔB 평균(CART 방식)이나 상위 주성분(Iso-C, DeVec 방식)을 공유 성분으로 추정한다.
  - ΔB_k에서 이 성분을 빼고 나서 지운다(Ethos/Ext-Sub 방식).
  - 예측: keep 손상은 줄고, decoy 손상은 거의 0이 되며, α\*는 1에 가까워진다.
  - LATA처럼 층별로 분석해 형식 성분이 어느 층에 몰리는지도 본다.
- **(c) α\* < 1의 원인을 구분한다.**
  - 원인 후보 1차 근사(Thudi식 전개에서 우리가 유도한 것, 문헌에 직접 있지는 않다): 최종 weight에 남은 k의 기여는 대략 Π_{j>k}(I − ηH_j)·ΔB_k다. 이후 write가 겹치는 방향을 다시 맞추면서 k의 흔적이 줄어든다.
  - 원인 후보 공유 성분: 공유 성분을 과하게 빼면 손상이 생겨 α를 낮추는 것이 이득이다.
  - 구분하는 법: α\*를 k 이후 write 수의 함수로 측정한다. 1차 근사가 맞다면 마지막 write에서 α\*≈1이다. 그리고 (b)를 적용한 뒤 α\*가 어떻게 바뀌는지 본다.
- **(d) Rehearsal 누출.**
  - 이후 write의 rehearsal에 k의 QA가 섞였다면, ΔB_{j>k}에도 k의 정보가 들어 있다. FedEraser가 말한 "남은 update에도 대상의 영향"과 같은 경우다.
  - 답 수준은 0%라도 확률 수준(Truth Ratio)과 leak@k에서 잔여가 남는지 확인한다.
- **(e) Relearn 속도.**
  - 빼기로 지운 모델, oracle, 원래 모델에 k의 QA를 몇 step 학습시켜 회복 곡선을 비교한다(Hu et al.; Deeb & Roger).
  - k와 무관한 사람으로 학습했을 때 k가 회복되는지도 본다(Łucki et al.).
- **(f) Paraphrase와 white-box probe.** paraphrase 질문, 그리고 hidden state를 logit lens로 본 정답 순위(Patil et al., 2023).
- **(g) 정확한 기준선의 비용.**
  - SISA/Rewind 방식: B_{k−1}로 되돌리고 write k+1..n을 replay.
  - 설계 대안: write별 직교 부분공간(O-LoRA, InfLoRA)이나 write별 slot.
  - 정확도·비용·keep 손상을 표로 비교한다.
- **(h) TMEM은 텍스트 explicit memory도 함께 쓴다.** fast weight만 지우면 텍스트 memory에서 다시 학습되는 backflow가 생길 수 있다(Agentic Unlearning, Wang et al., 2026). 평가 범위를 정할 때 명시한다.
- **(i) 측정 지표.** weight 거리(‖B_sub − B_oracle‖)만으로는 삭제를 증명할 수 없다(Thudi et al., 2022, forgeability). 행동 지표와 분포 지표를 같이 낸다.

---

## 5. 참고문헌

모든 항목은 arXiv API로 ID와 제목을 확인했다(제목 유사도 ≥ 0.70). venue는 arXiv 페이지에 있는 것과 검색 agent가 확인한 것만 적었다. 저자는 앞 3명 + et al.이다.

**TMEM과 parametric memory**
- Ren, T., Luo, W., Yang, H., et al. (2026). Scaling Self-Evolving Agents via Parametric Memory. arXiv:2606.04536.
- Das, P., Chaudhury, S., Nelson, E., et al. (2024). Larimar: Large Language Models with Episodic Memory Control. ICML 2024. arXiv:2403.11901.
- Behrouz, A., Zhong, P., Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time. arXiv:2501.00663.
- Lin, J., Zettlemoyer, L., Ghosh, G., et al. (2025). Continual Learning via Sparse Memory Finetuning. arXiv:2510.15103.
- Back, S., Lee, D., Kang, N., et al. (2026). Understanding LoRA as Knowledge Memory: An Empirical Analysis. arXiv:2603.01097.
- Wang, B., Wang, F., Wang, P., et al. (2026). Agentic Unlearning: When LLM Agent Meets Machine Unlearning. arXiv:2602.17692.

**저장한 update 빼기, 재계산, 선형화**
- Graves, L., Nagisetty, V., Ganesh, V. (2020). Amnesiac Machine Learning. AAAI 2021. arXiv:2010.10981.
- Wu, Y., Dobriban, E., Davidson, S. B. (2020). DeltaGrad: Rapid retraining of machine learning models. ICML 2020. arXiv:2006.14755.
- Thudi, A., Deza, G., Chandrasekaran, V., et al. (2021). Unrolling SGD: Understanding Factors Influencing Machine Unlearning. arXiv:2109.13398.
- Thudi, A., Jia, H., Shumailov, I., et al. (2021). On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning. USENIX Security 2022. arXiv:2110.11891.
- Liu, G., Ma, X., Yang, Y., et al. (2020). Federated Unlearning (FedEraser). IWQoS 2021. arXiv:2012.13891.
- Wu, C., Zhu, S., Mitra, P. (2022). Federated Unlearning with Knowledge Distillation. arXiv:2201.09441.
- Mu, S., Klabjan, D. (2024). Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions. arXiv:2409.09778.
- Hu, Y., Schölkopf, B., Sanyal, A. (2025). Online Learning and Unlearning. arXiv:2505.08557.
- Golatkar, A., Achille, A., Soatto, S. (2019). Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR 2020. arXiv:1911.04933.
- Golatkar, A., Achille, A., Soatto, S. (2020). Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations. ECCV 2020. arXiv:2003.02960.
- Golatkar, A., Achille, A., Ravichandran, A., et al. (2020). Mixed-Privacy Forgetting in Deep Networks. CVPR 2021. arXiv:2012.13431.
- Li, G., Hsu, H., Chen, C.-F., et al. (2023). Fast-NTK: Parameter-Efficient Unlearning for Large-Scale Models. arXiv:2312.14923.
- Guo, C., Goldstein, T., Hannun, A., et al. (2019). Certified Data Removal from Machine Learning Models. ICML 2020. arXiv:1911.03030.
- Neel, S., Roth, A., Sharifi-Malvajerdi, S. (2020). Descent-to-Delete: Gradient-Based Methods for Machine Unlearning. arXiv:2007.02923.

**Sharding, 격리, 직교 부분공간**
- Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., et al. (2019). Machine Unlearning. IEEE S&P 2021. arXiv:1912.03817.
- Fleshman, W., Khan, A., Marone, M., et al. (2024). AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees. arXiv:2404.08417.
- Hu, Z., Zhang, Y., Xiao, M., et al. (2024). Exact and Efficient Unlearning for Large Language Model-based Recommendation. arXiv:2404.10327.
- Kuo, K., Setlur, A., Srinivas, K., et al. (2025). Exact Unlearning of Finetuning Data via Model Merging at Scale. arXiv:2504.04626.
- Liu, Z., Kalinli, O. (2023). Forgetting Private Textual Sequences in Language Models via Leave-One-Out Ensemble. arXiv:2309.16082.
- Wang, X., Chen, T., Ge, Q., et al. (2023). Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA). Findings of EMNLP 2023. arXiv:2310.14152.
- Liang, Y.-S., Li, W.-J. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. CVPR 2024. arXiv:2404.00228.
- Gao, C., Wang, L., Ding, K., et al. (2024). On Large Language Model Continual Unlearning. ICLR 2025. arXiv:2407.10223.
- Rahulamathavan, Y., Iqbal, N., Hu, J., et al. (2026). Orthogonal Subspace Projection for Continual Machine Unlearning via SVD-Based LoRA. arXiv:2604.12526.
- Rachapudi, J., Vatsi, R., Hambarde, P., et al. (2026). BID-LoRA: A Parameter-Efficient Framework for Continual Learning and Unlearning. arXiv:2604.12686.
- Liu, X., Chang, X. (2025). LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning. CVPR 2025. arXiv:2503.18985.

**Task arithmetic, negation, 공유 성분**
- Ilharco, G., Ribeiro, M. T., Wortsman, M., et al. (2022). Editing Models with Task Arithmetic. ICLR 2023. arXiv:2212.04089.
- Ortiz-Jimenez, G., Favero, A., Frossard, P. (2023). Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models. NeurIPS 2023. arXiv:2305.12827.
- Jin, R., Hou, B., Xiao, J., et al. (2024). Fine-Tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic. ICLR 2025. arXiv:2407.07089.
- Zhang, J., Chen, S., Liu, J., et al. (2023). Composing Parameter-Efficient Modules with Arithmetic Operations. NeurIPS 2023. arXiv:2306.14870.
- Li, H., Zhang, Y., Zhang, S., et al. (2025). When is Task Vector Provably Effective for Model Editing? A Generalization Analysis of Nonlinear Transformers. ICLR 2025. arXiv:2504.10957.
- Gao, L., Niu, Y., Tang, T., et al. (2024). Ethos: Rectifying Language Models in Orthogonal Parameter Space. Findings of NAACL 2024. arXiv:2403.08994.
- Hu, X., Li, D., Hu, B., et al. (2023). Separate the Wheat from the Chaff: Model Deficiency Unlearning via Parameter-Efficient Module Operation. AAAI 2024. arXiv:2308.08090.
- Damirchi, H., Abbasnejad, E., Zhang, Z., et al. (2025). Decomposing Task Vectors for Refined Model Editing. arXiv:2512.22511.
- Chen, Y.-L., Wei, Y.-R., Hsu, C.-Y., et al. (2025). Layer-Aware Task Arithmetic: Disentangling Task-Specific and Instruction-Following Knowledge. arXiv:2502.20186.
- Kim, H. S., Han, D., Choe, J. (2024). NegMerge: Sign-Consensual Weight Merging for Machine Unlearning. ICML 2025. arXiv:2410.05583.
- Cai, C., Ye, Z., Yao, J., et al. (2026). Per-parameter Task Arithmetic for Unlearning in Large Language Models. arXiv:2601.22030.
- Yagnik, S., Gaur, S., Thakur, S., et al. (2026). MAAT: Multi-phase Adapter-Aware Targeted Unlearning. arXiv:2605.30514.
- Jeung, W., Yoon, S., Hong, H., et al. (2025). DUSK: Do Not Unlearn Shared Knowledge. arXiv:2505.15209.
- Ni, S., Chen, D., Li, C., et al. (2023). Forgetting before Learning: Utilizing Parametric Arithmetic for Knowledge Updating in Large Language Models. arXiv:2311.08011.
- Yadav, P., Tam, D., Choshen, L., et al. (2023). TIES-Merging: Resolving Interference When Merging Models. NeurIPS 2023. arXiv:2306.01708.
- Yu, L., Yu, B., Yu, H., et al. (2023). Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE). ICML 2024. arXiv:2311.03099.
- Wang, K., Dimitriadis, N., Ortiz-Jimenez, G., et al. (2024). Localizing Task Information for Improved Model Merging and Compression. ICML 2024. arXiv:2405.07813.
- Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., et al. (2024). Task Singular Vectors: Reducing Task Interference in Model Merging. CVPR 2025. arXiv:2412.00081.
- Marczak, D., Magistri, S., Cygert, S., et al. (2025). No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces. ICML 2025. arXiv:2502.04959.
- Lu, Z., Fan, C., Wei, W., et al. (2024). Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging. NeurIPS 2024. arXiv:2406.15479.
- Choi, J., Kim, D., Lee, C., et al. (2024). Revisiting Weight Averaging for Model Merging (CART). arXiv:2412.12153.
- Xiong, F., Cheng, R., Chen, W., et al. (2024). Multi-Task Model Merging via Adaptive Weight Disentanglement. arXiv:2411.18729.
- Shuttleworth, R., Andreas, J., Torralba, A., et al. (2024). LoRA vs Full Fine-tuning: An Illusion of Equivalence. arXiv:2410.21228.
- Zheng, J., Cai, X., Qiu, S., et al. (2025). Spurious Forgetting in Continual Learning of Language Models. ICLR 2025. arXiv:2501.13453.
- Jiang, G., Li, Z., Lian, D., et al. (2024). Refine Large Language Model Fine-tuning via Instruction Vector. arXiv:2406.12227.
- Huang, S.-C., Li, P.-Z., Hsu, Y.-C., et al. (2023). Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages. ACL 2024. arXiv:2310.04799.
- Yin, H., Wang, H., Luo, Y., et al. (2026). ChainLoRA: Geometry-Preserving Task Vector Merging for Continual Learning in LLMs. arXiv:2610.00431.

**Model editing과 지식 저장**
- Meng, K., Bau, D., Andonian, A., et al. (2022). Locating and Editing Factual Associations in GPT (ROME). NeurIPS 2022. arXiv:2202.05262.
- Meng, K., Sen Sharma, A., Andonian, A., et al. (2022). Mass-Editing Memory in a Transformer (MEMIT). ICLR 2023. arXiv:2210.07229.
- Hartvigsen, T., Sankaranarayanan, S., Palangi, H., et al. (2022). Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors. NeurIPS 2023. arXiv:2211.11031.
- Yu, L., Chen, Q., Zhou, J., et al. (2023). MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA. AAAI 2024. arXiv:2312.11795.
- Wang, P., Li, Z., Zhang, N., et al. (2024). WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models. NeurIPS 2024. arXiv:2405.14768.
- Fang, J., Jiang, H., Wang, K., et al. (2024). AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. ICLR 2025. arXiv:2410.02355.
- Huang, Z., Shen, Y., Zhang, X., et al. (2023). Transformer-Patcher: One Mistake worth One Neuron. ICLR 2023. arXiv:2301.09785.
- Gupta, A., Rao, A., Anumanchipalli, G. (2024). Model Editing at Scale leads to Gradual and Catastrophic Forgetting. Findings of ACL 2024. arXiv:2401.07453.
- Hu, C., Cao, P., Chen, Y., et al. (2024). Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models. AAAI 2025. arXiv:2408.07413.
- Youssef, P., Zhao, Z., Schlötterer, J., et al. (2024). How to Make LLMs Forget: On Reversing In-Context Knowledge Edits. NAACL 2025. arXiv:2410.12586.
- Youssef, P., Zhao, Z., Seifert, C., et al. (2025). Tracing and Reversing Edits in LLMs. arXiv:2505.20819.
- Cheng, Y., Youssef, P., Seifert, C., et al. (2025). Can Fine-Tuning Erase Your Edits? On the Fragile Coexistence of Knowledge Editing and Adaptation. arXiv:2511.05852.
- Patil, V., Hase, P., Bansal, M. (2023). Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks. ICLR 2024. arXiv:2309.17410.
- Li, Z., Wang, X., Shen, W. F., et al. (2025). Editing as Unlearning: Are Knowledge Editing Methods Strong Baselines for Large Language Model Unlearning? arXiv:2505.19855.
- Gandikota, R., Feucht, S., Marks, S., et al. (2024). Erasing Conceptual Knowledge from Language Models. arXiv:2410.02760.
- Cohen, R., Biran, E., Yoran, O., et al. (2023). Evaluating the Ripple Effects of Knowledge Editing in Language Models. TACL 2024. arXiv:2307.12976.
- Li, Z., Zhang, N., Yao, Y., et al. (2023). Unveiling the Pitfalls of Knowledge Editing for Large Language Models. ICLR 2024. arXiv:2310.02129.
- Zhang, Y., Li, S., Qian, C., et al. (2025). The Law of Knowledge Overshadowing: Towards Understanding, Predicting, and Preventing LLM Hallucination. arXiv:2502.16143.
- Sun, C., Aksitov, R., Zhmoginov, A., et al. (2025). How new data permeates LLM knowledge and how to dilute it. arXiv:2504.09522.
- Gekhman, Z., Yona, G., Aharoni, R., et al. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? EMNLP 2024. arXiv:2405.05904.
- Kang, K., Wallace, E., Tomlin, C., et al. (2024). Unfamiliar Finetuning Examples Control How Language Models Hallucinate. arXiv:2403.05612.
- Ghosal, G., Hashimoto, T., Raghunathan, A. (2024). Understanding Finetuning for Factual Knowledge Extraction. ICML 2024. arXiv:2406.14785.
- Allen-Zhu, Z., Li, Y. (2023). Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. arXiv:2309.14316.
- Pletenev, S., Marina, M., Moskovskiy, D., et al. (2025). How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM? arXiv:2502.14502.
- Zhou, C., Liu, P., Xu, P., et al. (2023). LIMA: Less Is More for Alignment. NeurIPS 2023. arXiv:2305.11206.
- Kung, P.-N., Peng, N. (2023). Do Models Really Learn to Follow Instructions? An Empirical Study of Instruction Tuning. ACL 2023. arXiv:2305.11383.

**LLM unlearning과 평가**
- Jang, J., Yoon, D., Yang, S., et al. (2022). Knowledge Unlearning for Mitigating Privacy Risks in Language Models. arXiv:2210.01504.
- Maini, P., Feng, Z., Schwarzschild, A., et al. (2024). TOFU: A Task of Fictitious Unlearning for LLMs. arXiv:2401.06121.
- Zhang, R., Lin, L., Bai, Y., et al. (2024). Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning. arXiv:2404.05868.
- Shi, W., Lee, J., Huang, Y., et al. (2024). MUSE: Machine Unlearning Six-Way Evaluation for Language Models. arXiv:2407.06460.
- Li, N., Pan, A., Gopal, A., et al. (2024). The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning. arXiv:2403.03218.
- Eldan, R., Russinovich, M. (2023). Who's Harry Potter? Approximate Unlearning in LLMs. arXiv:2310.02238.
- Yao, Y., Xu, X., Liu, Y. (2023). Large Language Model Unlearning. arXiv:2310.10683.
- Liu, S., Yao, Y., Jia, J., et al. (2024). Rethinking Machine Unlearning for Large Language Models. Nature Machine Intelligence. arXiv:2402.08787.
- Fan, C., Liu, J., Lin, L., et al. (2024). Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning. arXiv:2410.07163.
- Mekala, A., Dorna, V., Dubey, S., et al. (2024). Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models. COLING 2025. arXiv:2409.13474.
- Yuan, X., Pang, T., Du, C., et al. (2024). A Closer Look at Machine Unlearning for Large Language Models. ICLR 2025. arXiv:2410.08109.
- Gu, R., Du, J., Zhang, Y., et al. (2026). Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning. arXiv:2605.08765.
- Jin, Z., Cao, P., Wang, C., et al. (2024). RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models. arXiv:2406.10890.
- Ye, X., Zhang, M., Wu, S. (2025). LLM Unlearning Should Be Form-Independent. arXiv:2506.07795.
- Hu, S., Fu, Y., Wu, Z. S., et al. (2024). Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning. ICLR 2025. arXiv:2406.13356.
- Łucki, J., Wei, B., Huang, Y., et al. (2024). An Adversarial Perspective on Machine Unlearning for AI Safety. TMLR. arXiv:2409.18025.
- Lynch, A., Guo, P., Ewart, A., et al. (2024). Eight Methods to Evaluate Robust Unlearning in LLMs. arXiv:2402.16835.
- Zhang, Z., Wang, F., Li, X., et al. (2024). Catastrophic Failure of LLM Unlearning via Quantization. ICLR 2025. arXiv:2410.16454.
- Deeb, A., Roger, F. (2024). Do Unlearning Methods Remove Information from Language Model Weights? arXiv:2410.08827.
- Lee, B. W., Foote, A., Infanger, A., et al. (2025). Distillation Robustifies Unlearning. arXiv:2506.06278.
- Fan, C., Jia, J., Zhang, Y., et al. (2025). Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond. ICML 2025. arXiv:2502.05374.
- Cooper, A. F., Choquette-Choo, C. A., Bogen, M., et al. (2024). Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research. arXiv:2412.06966.
- Xu, X., Yue, X., Liu, Y., et al. (2025). Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs. arXiv:2505.16831.
- Sun, G., Manakul, P., Zhan, X., et al. (2025). Unlearning vs. Obfuscation: Are We Truly Removing Knowledge? EMNLP 2025. arXiv:2505.02884.
- Thaker, P., Hu, S., Kale, N., et al. (2024). Position: LLM Unlearning Benchmarks are Weak Measures of Progress. arXiv:2410.02879.
- Dorna, V., Mekala, A., Zhao, W., et al. (2025). OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics. arXiv:2506.12618.
- Cho, S., Hwang, D., Sala, F., et al. (2025). Reference-Specific Unlearning Metrics Can Hide the Truth: A Reality Check. arXiv:2510.12981.
- Yoon, S., Jun, Y., No, A. (2026). Position: The Term "Machine Unlearning" Is Overused in LLMs. arXiv:2606.27379.
- Reisizadeh, H., Ruan, J., Chen, Y., et al. (2025). Leak@k: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding. arXiv:2511.04934.
- Ko, M., Just, H. A., Fleming, C., et al. (2025). Probing Knowledge Holes in Unlearned LLMs. arXiv:2511.00030.

---

## 6. 확인하지 못한 항목 (참고문헌에서 제외)
- "Separating Common and Unique Directions for Model Merging" (CUSD, NeurIPS 2026 poster): arXiv ID를 찾지 못했다. common bulk와 unique tail로 나눈다는 설명이 있다.
- "Reversibility of parametric knowledge editing" (Applied Sciences 2026, 16, 6567): 검색 snippet만 봤다.
- PALoRA (arXiv:2605.24549?): skill 부분공간과 직교하게 사실을 주입한다. snippet만 봤고 ID를 확인하지 못했다.
- MEMIT-Merge (arXiv:2502.07322?): snippet만 봤다.
- M+ (MemoryLLM 후속), MemoryAgentBench, "Do Self-Evolving Agents Forget?" (arXiv:2605.09315?): abstract를 확인하지 못했다.
- 참고문헌의 저자 목록은 모두 arXiv API가 돌려준 앞 3명으로 맞췄다(§5).

## 7. 한계
- 체계적 검색(PRISMA)이 아니다. "decoy 대조군이 선행 연구에 없다"는 주장은 이번 검색 범위에서만 성립한다.
- 대부분은 abstract와 HTML 일부를 읽고 요약했다. 본문까지 확인한 것: TOFU, Ilharco et al. 부록, Zhang et al. 2023 부록(λ 값), TMEM 설정, Amnesiac ML, Thudi et al.
- MUSE의 Task Vector 수치는 원문 표에서 다시 확인해야 한다.
- AI 사용 고지: 이 문서는 Claude(ARS deep-research 스킬, 하위 검색 agent 4개)로 작성했다. 인용 존재 여부는 arXiv API로 자동 확인했다.
