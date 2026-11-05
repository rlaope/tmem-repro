# 주장 5개의 선행 연구 확인 (novelty check, 2026-11-05)

**범위.** 첫 조사(`related-work-ars-20261105.md`, arXiv만 대상, 참고문헌 114개)를 반복하지 않는다. 이번에는 arXiv 밖을 찾았다.
- OpenReview API 전문 검색: ICLR 2027 제출작, ICML/NeurIPS 2025–2026과 workshop, ACL ARR 포함
- ACL Anthology
- 웹 검색(Google Scholar 성격의 질의)

Semantic Scholar API는 rate limit(429)에 걸려 쓰지 못했다.

**대상 설정.**
- TMEM식 공유 LoRA를 쓴다: SVD로 초기화한 A는 고정하고, B만 plain SGD로 학습한다.
- write 하나는 가상 인물 한 명의 QA로 SGD를 몇 epoch 돌리는 것이다.
- write k는 기록해 둔 ΔB_k를 빼서 지운다. 비교 기준은 k 없이 재학습한 oracle이다.

**판정 기준.**
- ALREADY DONE: 같은 연산과 같은 평가를 한 연구가 있다.
- PARTIALLY DONE: 핵심 아이디어나 연산은 있지만 설정·대상·평가 중 하나 이상이 다르다.
- NOT FOUND: 이번 검색 범위에서 찾지 못했다.

---

## 요약 표

| # | 주장 | 가장 가까운 선행 연구 | 판정 | 차이점 |
|---|---|---|---|---|
| 1 | 온라인으로 누적한 공유 LoRA/fast-weight agent memory에서, 기록한 update를 빼서 write 하나를 지우고, k 없이 재학습한 oracle과 비교 | **Amnesiac ML** (Graves et al., 2021; 이미 첫 조사에 있음). **VLA task-vector negation audit** (Wang et al., 2026): skill별 task vector를 빼고 leave-one-task-out 재학습 oracle과 비교했다(target 하나). **MemEdit** (ICLR 2027 제출): parametric agent memory에 write마다 가역 low-rank delta를 journal로 남긴다. **RCMF** (ICLR 2027 제출), **Memory Adapters** (Grimes et al., ICML 2026 MemFM WS) | **PARTIALLY DONE** | 연산(저장한 update 빼기)과 oracle 비교는 따로따로 존재한다. 그러나 어느 연구도 (a) 에이전트가 온라인으로 쓰는 *하나의 공유* LoRA에서, (b) plain SGD write의 ΔB를 빼서, (c) 재학습 oracle과 비교하지 않았다. MemEdit, RCMF, Memory Adapters는 기록 단위로 격리하거나 null-space 제약을 거는 설계여서 공유 성분 문제를 처음부터 피한다. MemEdit의 Delete는 delta를 빼는 방식이 아니라 projected gradient ascent다. VLA audit은 오프라인 multitask expert(공유 init에서 skill별 LoRA/full FT)이고, 누적된 하나의 adapter가 아니다. |
| 2 | Decoy 대조군: 같은 모델 상태에서 학습한 무관한 item의 update를 빼서, keep 손상이 item에 고유한지 시험 | **"Unlearning Can Hurt When There Is Nothing to Forget"** (ICLR 2027 제출, 익명). forget 문서를 학습한 사본과 무관한 문서를 학습한 사본에 같은 RMU/NPO를 돌렸더니 손상이 거의 같았다(RMU의 MMLU 부수 손상 중 74% 이상이 내용과 무관). **PLACEBU** (ICLR 2027 제출): target과 placebo를 짝지은 대조. **VLA audit**: 10×10 교차 빼기 행렬. **Ilharco et al. 2023**: random vector 대조군 | **PARTIALLY DONE** | "부수 손상이 지운 내용에 고유하지 않다"는 *개념*과 그것을 재는 대조 실험은 이미 있다. 그중 "Nothing to Forget"이 가장 가깝고, 논문에 반드시 인용해야 한다. 차이점: 그 논문은 *모델*을 바꾸고(학습 안 한 사본) gradient 기반 unlearning을 그대로 돌린다. 우리는 *빼는 update*를 바꾼다(같은 상태에서 학습한 decoy의 ΔB). "저장한 update 빼기"에 decoy-update 대조군을 쓴 연구는 찾지 못했다. VLA audit의 교차 행렬도 비대상 skill을 빼긴 하지만 decoy 대조로 설계·해석하지 않았다. |
| 3 | 같은 상태에서 decoy를 학습해 공유 성분을 추정하고, ΔB_k 중 decoy update와 직교하는 부분만 뺀다(decoy 방향의 투영은 남긴다) | **Ext-Sub** (Hu et al., AAAI 2024): LoRA 행마다 anti-expert 벡터를 expert와 anti-expert 단위벡터의 합 방향에 투영해 "general capability"로 두고, 나머지만 뺀다. **UNLEARN** (Lizzo & Heck, Findings of NAACL 2025): target task 행렬의 특이벡터에서 *유사 task*의 특이벡터 성분을 Gram-Schmidt로 뺀 뒤, 남은 T′를 W에서 뺀다. **PLACEBU**: placebo로 공유 효과와 forget 고유 효과를 가른다. **Ethos**(첫 조사에 있음) | **PARTIALLY DONE** | "기준 update에 대한 투영을 남기고 직교 성분만 뺀다"는 연산은 Ext-Sub와 UNLEARN에 거의 그대로 있다. 차이점: (a) 기준이 *보존해야 할* expert나 유사 task다. 우리는 공유(형식·정렬) 성분만 추정하려고 일부러 만든 *무관한 decoy*다. (b) decoy를 *삭제 대상과 같은 모델 상태*에서 학습해 그 상태의 공통 성분을 추정한다. (c) 온라인 누적 memory에서 하고, 재학습 oracle로 평가한다. 투영 단위가 행별(Ext-Sub)인지, 특이벡터별(UNLEARN)인지, 전체 flatten인지도 구분해서 밝혀야 한다. |
| 4 | 각 write가 이전 write의 QA를 rehearsal하므로, 지울 item이 이후 rehearsal에도 들어 있을 때의 삭제 | **FedEraser** (Liu et al., 2021; 첫 조사): 남은 client update에 대상의 영향이 섞여 있다. **CLPU** (Liu, Liu & Stone, CoLLAs 2022): 지울 수 있는 task는 별도 모델로 격리하고 replay(DER++)에 섞지 않아 정확한 삭제를 얻는다. **UnCLe** (2025): 지운 task가 이후 학습에서 되살아나는 "task relapse". **Write-Back Redundancy** (ARR 2026 제출): 에이전트가 쓴 near-duplicate 기록 때문에 leave-one-out 삭제가 실패한다(retrieval memory) | **PARTIALLY DONE** (문제 인식 수준) | "지운 item의 영향이 이후 update/기록에 남는다"는 문제는 연합학습, continual learning, retrieval agent memory에서 각각 알려져 있다. 그러나 (a) parametric memory에서 rehearsal이 삭제 대상을 이후 write에 다시 주입하는 경우에, (b) 저장한 update 빼기로 지우고, (c) rehearsal에서도 k를 뺀 재학습 oracle과 비교해 잔여를 정량화한 연구는 찾지 못했다. CLPU는 이 문제를 격리로 *회피*하고, 측정하지 않는다. |
| 5 | 관찰: 다른 write update들의 평균을 투영해서 빼면 손상이 줄지 않는데, 같은 상태의 decoy 방향을 투영해서 빼면 줄어든다 | **CART**, **Iso-C**, **DeVec**(첫 조사): 공유 성분을 task vector 평균이나 공통 부분공간으로 정의한다. **VLA audit**: retained-skill 부분공간에 직교화한 negation(retain-orth)과 서로 직교화한 negation(mutual-orth)이 다중 skill 선택성을 "rescue하지 못한다". **Ext-Sub/UNLEARN**: 기준 update 하나에 대한 투영 | **NOT FOUND** | 두 공유 성분 추정치를 같은 조건에서 비교한 연구는 없었다. 한쪽은 다른 write의 평균이고 다른 쪽은 같은 상태의 decoy다. 평균이나 공통 부분공간이 공유 성분이라는 가정은 merging 쪽에서 흔하다. VLA audit의 음성 결과는 우리 결과의 "평균은 안 된다" 쪽과 방향이 맞지만, 설정이 다르고(retain 부분공간, 다중 negation) decoy와 비교하지 않았다. 해석상 이 주장은 "공유 성분은 상태에 따라 달라진다(state-dependent)"는 진술이 된다. 이것이 가장 새로운 부분으로 보인다. |

---

## 주장별 근거

### 주장 1: 공유 LoRA에서 기록한 update를 빼서 지우고 oracle과 비교
- **Amnesiac ML**(첫 조사 §1.1)은 batch update를 저장해 두고 빼는 방법이고, 재학습과 같지 않다는 점(path dependence)도 그 논문 스스로 밝혔다. 따라서 *방법*은 새롭지 않다.
- **Wang et al. 2026 (VLA audit, arXiv:2608.04692).** LIBERO-Goal 10개 skill에서 per-skill task vector τ_i = θ_exp,i − θ_base를 뺐다.
  - expert는 공유 초기값에서 rank-64 LoRA로 학습했다. π0.5 실험은 full FT다.
  - Table 2에 skill t0 하나에 대한 "leave-one-task-out retraining oracle"이 있다(seed 1개).
  - 결과: 5개 skill은 선택적으로 억제됐고, 3개는 저항했고, 2개는 전역 붕괴했다. control 유지율은 평균 약 52%다. relearning probe를 보면 행동을 가린 것(masking)에 가깝다.
  - 우리와 다른 점: 오프라인 multitask expert이고, 온라인으로 누적한 adapter 하나가 아니다. oracle 비교도 target 하나에 한정된다.
- **MemEdit (ICLR 2027 제출, OpenReview 5QSABMAdjf).** parametric agent memory에 CRUD를 제공하는 프레임워크다.
  - write는 null-space 제약을 건 rank-one update이고, "journaled as a reversible low-rank delta"로 기록된다.
  - Delete는 projected gradient ascent다.
  - 설계 자체가 기록 사이 간섭을 막는다. 공유 성분 문제를 회피하는 쪽이다.
- **RCMF (ICLR 2027 제출, lC1U96Au7V).** 결정 transition마다 따로 parametric 기여를 두고 additive하게 넣고 뺀다. 공유 parameter에 흡수된 경험은 "selectively remove하기 어렵다"고 명시하고, 그래서 격리를 택한 설계다.
- **Memory Adapters (Grimes et al., ICML 2026 MemFM WS oral, LCVFastmwR).** sequence 단위 gradient update를 작은 모듈로 격리해 TOFU에서 즉시 정확하게 삭제한다.
- **TMEM 원 논문**(arXiv:2606.04536; OpenReview에는 ICLR 2027 desk reject로 올라 있다)은 삭제를 다루지 않는다.
- **결론.** "격리로 정확한 삭제"는 이미 많이 했다. "공유 SGD LoRA에서 빼기로 근사 삭제하고 oracle로 감사(audit)"는 우리 설정만의 조합이다. 논문에서는 *설정과 감사*를 기여로 내세우고, *방법*을 기여로 내세우지 않는 것이 맞다.

### 주장 2: decoy 대조군
- **"Unlearning Can Hurt When There Is Nothing to Forget" (ICLR 2027 제출, bJ4hT11duP).** 가장 가까운 연구이고, 대조군의 개념적 새로움을 상당 부분 선점했다.
  - 같은 방식으로 두 사본을 fine-tune한다. 하나는 forget 문서로, 하나는 무관한 문서로 학습한다. 둘에 똑같은 unlearning run을 돌린다.
  - RMU에서는 학습하지 않은 사본도 거의 같은 손상을 입었다. MMLU 부수 손상의 74% 이상, 나머지 두 데이터셋에서는 90% 이상이 내용과 무관했다.
  - GradDiff와 SimNPO에서는 의존성이 검출됐다.
  - 저자들은 "collateral damage를 never-learned 사본의 손상과 나란히 보고하라"고 권고한다.
  - 우리와 다른 점: 우리 decoy는 *빼는 벡터*를 바꾸는 대조다. 그 논문은 *모델 이력*을 바꾸는 대조다. 우리 경우 "무엇을 빼든 비슷하게 망가진다"는 결과가 *update 빼기*라는 연산에 대해 같은 결론을 낸다.
- **PLACEBU (ICLR 2027 제출, 6zLThckmMt).** forget 예제 주변을 구조적으로 perturb해 "matched target-placebo controls"를 만들고, forget 고유 효과와 공유 효과를 가른다(retain-free). 주장 3의 철학과도 가깝다.
- **VLA audit.** 10×10 교차 빼기 행렬로 어떤 source skill을 뺐을 때 어떤 control이 무너지는지 본다(interference graph, t3이 hub). decoy 대조로 설계하지는 않았다.
- **Ilharco et al.** norm을 맞춘 random vector만 썼다.
- **결론.** "decoy 대조군은 새로운 진단"이라는 첫 조사의 표현은 약하게 고쳐야 한다. 쓸 수 있는 표현은 "update-subtraction 삭제에 맞춘 decoy-update 대조군"이다. "Nothing to Forget"의 never-learned 대조를 *update 빼기* 쪽으로 옮긴 것으로 위치를 잡고 인용한다.

### 주장 3: decoy 방향을 남기고 직교 성분만 빼기
- **Ext-Sub.** 원문 Algorithm 1을 직접 확인했다.
  - LoRA ΔW의 각 행 v_i에 대해 다음을 계산한다.
    - v_i^∘ = v_i⁺/|v_i⁺| + v_i⁻/|v_i⁻|
    - general = v_i⁻ 를 v_i^∘ 에 투영한 것
    - Ext(v_i⁻) = v_i⁻ − general
    - θ′ = θ⁺ − λ·Ext(θ⁻)
  - 즉 "기준 방향에 대한 투영은 남기고 직교 성분만 뺀다"와 같은 구조다. 기준은 *expert*(보존 대상)와 anti-expert의 이등분선이다.
- **UNLEARN.** 원문 §3.1–3.2의 HTML을 확인했다.
  - layer별 task 행렬 T_i를 bottleneck(rank 16)으로 학습한다.
  - T_i의 특이벡터에서 유사 task T_o의 특이벡터 성분을 Gram-Schmidt로 제거한다.
  - SVD로 T_i′를 재구성한 뒤 W′ − T_i′를 계산한다.
  - "저장한 update에서 유사 task 성분을 투영으로 제거한 뒤 빼기"의 직접 선례다. 보고 수치: 유사 task에서 target 망각 약 80%, 유사 task 성능은 원본 대비 10% 이내 유지.
- **차이점과 새로운 부분.**
  - 기준 update를 *보존할 지식*이 아니라 *무관한 decoy*로 둔다.
  - 그 decoy를 *삭제 대상과 같은 모델 상태*에서 학습해 그 상태의 공통(형식/정렬) 성분 추정치로 쓴다.
  - 온라인 누적 memory에서 재학습 oracle로 평가한다.
- **결론.** 연산 자체는 PARTIALLY DONE이다. "same-state decoy를 공유 성분 추정기로 쓴다"는 점은 이번 검색 범위에서 찾지 못했다.

### 주장 4: rehearsal이 있을 때의 삭제
- **FedEraser.** 남은 client의 update에 대상의 영향이 섞여 있어 calibration이 필요하다. 구조적으로 가장 비슷한 선례다.
- **CLPU.** "temporarily learned" task를 별도 모델로 격리하고, 영구화하기 전까지 메인 모델과 replay에 넣지 않아 exact private unlearning을 얻는다. 우리 설정을 정확하게 만드는 설계 대안이다(rehearsal에서 삭제 가능한 write를 제외).
- **UnCLe (arXiv:2509.17530).** continual learning에서 이미 지운 task가 이후 학습에서 되살아나는 task relapse를 다룬다. abstract만 확인했다. replay buffer를 쓰는지는 확인하지 못했다.
- **Write-Back Redundancy (ARR 2026 Oct 제출, gSBEPUmVk2).**
  - 에이전트가 자기 출력을 memory에 다시 써서 near-duplicate 계보(lineage)가 생기면, 기록 하나를 빼고 replay하는 방식의 leave-one-out 삭제·귀속이 실패한다.
  - 기록 하나만 남아도 공격 성공률이 0.79다.
  - retrieval memory이지만 "삭제 대상의 복제본이 이후 기록에 퍼진다"는 구조가 우리 rehearsal과 같다.
- **결론.** 문제는 인식되어 있다(PARTIALLY DONE). parametric rehearsal 설정에서 update 빼기의 잔여를 oracle 기준으로 잰 연구는 없다. 첫 조사 §4.3(d)의 Truth Ratio와 leak@k 측정이 이 차이를 채운다.

### 주장 5: 다른 write 평균의 투영은 효과가 없고, same-state decoy의 투영은 효과가 있다
- merging 쪽(CART, Iso-C, DeVec)은 평균이나 공통 부분공간을 공유 성분으로 쓰는 것을 *기본 가정*으로 둔다.
- VLA audit에서는 retain 부분공간에 직교화한 negation과 서로 직교화한 negation이 선택성을 rescue하지 못했다. 단, 실험은 {t0, t2} 쌍 하나뿐이고, 부분공간을 정확히 어떻게 정의했는지 본문에 명시되지 않았다.
- 같은 조건에서 "다른 write의 평균"과 "같은 상태의 decoy"를 비교해, 공유 성분이 *상태 의존적*이라서 후자만 효과가 있다고 보인 연구는 찾지 못했다(NOT FOUND).
- 논문에서 가장 내세울 만한 관찰이다. 다만 비교가 공정했는지는 함께 보고해야 한다. 점검할 것: 평균 벡터와 decoy 벡터의 norm·rank가 같은지, 평균에 k 이후 write가 포함되는지, 투영 단위(행/전체).

---

## 첫 조사 문서에 반영할 수정
1. §2와 §4.2-2에 있는 "decoy 대조군을 찾지 못했다"는 문장을 고친다. never-learned 대조("Nothing to Forget")와 placebo 대조(PLACEBU)가 있다. 우리 기여는 "update 빼기용 decoy-update 대조"로 좁힌다.
2. 공유 성분 투영(§4.3 b)의 선례에 UNLEARN(Gram-Schmidt 후 빼기)을 넣는다. Ext-Sub 설명은 "행별 이등분선 투영"으로 구체화한다.
3. 설정 선례로 VLA audit을 넣는다. task-vector 빼기를 leave-one-out oracle로 감사했고, relearning probe를 썼다.

## 한계
- 체계적 검색이 아니다. OpenReview API 검색과 웹 검색을 합쳐 질의 약 30개를 돌렸다. Semantic Scholar는 429로 쓰지 못했고, Google Scholar는 직접 크롤링하지 않고 웹 검색으로 대신했다.
- 본문까지 확인한 것: Ext-Sub의 방법 절(PDF 3–4쪽), UNLEARN의 방법 절(HTML), VLA audit의 일부(HTML 앞 20만 자).
- 나머지는 abstract만 읽었다. 특히 "Nothing to Forget", PLACEBU, MemEdit, RCMF, Write-Back Redundancy가 그렇다. 이 다섯 편은 동료 심사를 거치지 않은 익명 제출작이다(ICLR 2027, ARR). 인용할 때 "under review"로 표기하고, 최종본 수치를 다시 확인해야 한다.
- UnCLe가 replay를 쓰는지는 확인하지 못했다.
- AI 사용 고지: Claude가 검색하고 작성했다. 인용 항목은 모두 실제 페이지(arXiv, OpenReview, ACL Anthology, PMLR)에서 제목과 저자를 확인했다.

---

## 참고문헌 (이번에 새로 찾은 것 + 판정에 직접 쓴 기존 항목)

**새로 찾은 것**
- Anonymous (2026). Unlearning Can Hurt When There Is Nothing to Forget. ICLR 2027 Conference Submission (under review). https://openreview.net/forum?id=bJ4hT11duP
- Anonymous (2026). PLACEBU: Placebo-Controlled Retain-Free LLM Unlearning. ICLR 2027 Conference Submission (under review). https://openreview.net/forum?id=6zLThckmMt
- Wang, S., Guo, W., Dai, R., Zhao, Y., Guo, Y., Xiong, H. (2026). Suppression Sticks, Locality Is Fragile: A Closed-Loop Target-and-Control Audit of Task-Vector Negation in VLA Policies. arXiv:2608.04692. https://arxiv.org/abs/2608.04692
- Lizzo, T., Heck, L. (2025). UNLEARN: Efficient Removal of Knowledge in Large Language Models. Findings of NAACL 2025. https://aclanthology.org/2025.findings-naacl.405 (arXiv:2408.04140)
- Anonymous (2026). MemEdit: Fine-Grained Memory Surgery for Expandable Parametric Agent Memory. ICLR 2027 Conference Submission (under review). https://openreview.net/forum?id=5QSABMAdjf
- Anonymous (2026). Transition-Parametric Memory for Language Agents: Reversible Compiled Memory Fields. ICLR 2027 Conference Submission (under review). https://openreview.net/forum?id=lC1U96Au7V
- Grimes, K., Kuo, K., Wu, S., Smith, V., Connor, M. C. (2026). Memory Adapters Enable Fast, Flexible Knowledge Unlearning in LLMs. ICML 2026 MemFM Workshop (oral). https://openreview.net/forum?id=LCVFastmwR
- Anonymous (2026). Write-Back Redundancy Breaks Leave-One-Out Attribution in Self-Evolving Agent Memory. ACL ARR 2026 October Submission. https://openreview.net/forum?id=gSBEPUmVk2
- Liu, B., Liu, Q., Stone, P. (2022). Continual Learning and Private Unlearning. CoLLAs 2022, PMLR 199:243–254. https://proceedings.mlr.press/v199/liu22a.html (arXiv:2203.12817)
- UnCLe: An Unlearning Framework for Continual Learning (2025). arXiv:2509.17530. https://arxiv.org/abs/2509.17530 (저자 미확인. abstract만 읽음)
- Wuerkaixi, A., Wang, Q., Cui, S., et al. (2025). Adaptive Localization of Knowledge Negation for Continual LLM Unlearning (ALKN). ICML 2025. https://openreview.net/forum?id=zEonUm7DTp (참고: task-vector 기반 연속 unlearning. 판정에는 직접 쓰지 않음)

**판정에 쓴 기존 항목(상세는 첫 조사 §5)**
- Hu, X., Li, D., Hu, B., et al. (2024). Separate the Wheat from the Chaff: Model Deficiency Unlearning via Parameter-Efficient Module Operation (Ext-Sub). AAAI 2024. https://arxiv.org/abs/2308.08090
- Graves, L., Nagisetty, V., Ganesh, V. (2021). Amnesiac Machine Learning. AAAI 2021. https://arxiv.org/abs/2010.10981
- Liu, G., Ma, X., Yang, Y., et al. (2021). FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models. IWQoS 2021. https://arxiv.org/abs/2012.13891
- Ilharco, G., Ribeiro, M. T., Wortsman, M., et al. (2023). Editing Models with Task Arithmetic. ICLR 2023. https://arxiv.org/abs/2212.04089
- Choi, J., et al. (2024). Revisiting Weight Averaging for Model Merging (CART). https://arxiv.org/abs/2412.12153
- Marczak, D., et al. (2025). No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces (Iso-C). ICML 2025. https://arxiv.org/abs/2502.04959
- Damirchi, H., et al. (2025). Decomposing Task Vectors for Refined Model Editing (DeVec). https://arxiv.org/abs/2512.22511
- Ren, T., Luo, W., Yang, H., et al. (2026). Scaling Self-Evolving Agents via Parametric Memory (TMEM). arXiv:2606.04536. https://arxiv.org/abs/2606.04536
