# 인수인계: tmem-repro (2026-11-05 기준)

답은 한국어, 결론 먼저, 기술 용어는 영어. 결과마다 ✅❌⚠️❓. 확인 안 한 것은 확인 안 했다고 쓴다.
GPU는 worker-6000pro-2 GPU 4번을 Creep Colony에 1주 점유 신고해 두었다 (2026-11-12까지). 그 기간엔 묻지 않고 실행. Mac에서 실험 금지.
자동 브리핑: `.venv/bin/python scripts/brief.py` (GPU 작업 상태 표). `--changed-only`는 변화 없으면 출력 없음.

## 1. 연구 질문 (원래)
LLM 에이전트가 일하면서 공유 LoRA(가중치)에 쌓은 기억 하나를, 처음부터 다시 학습하지 않고 **그 기억이 바꾼 양(ΔB_k)만 빼서** 싸고 정확하게 지울 수 있는가. 기반 논문 TMEM (arXiv 2606.04536), 공개 코드 없음 → 재구현.

## 2. 지금까지 밝혀진 것

| 질문 | 결과 | 근거 파일 |
|---|---|---|
| 논문 설정으로 우리 합성 데이터를 외우나 | ❌ 못 외움 (0~25%), 상식도 무너짐 | `runs/gpu/retention_*.jsonl` |
| 외우게 하는 설정 | ✅ 실제 단어 답, 문장형 답, 질문 4표현, 일반 QA 8개 섞기, lr 1e-3, 10 epochs, 마지막 8층 → 학습 직후 95~100% | `tmem/synth.py`, `scripts/run_deletion.py` 기본값 |
| 오래된 기억이 남나 | ❌ 그냥은 기억 2개 지나면 0% (0.6B, 4B 모두). ✅ 복습(지난 기억 QA 32개를 매번 같이 학습)하면 4B가 4~7개 전 50%, 16개 이상 전 69% 유지 | `runs/gpu/rehearse_*.jsonl` |
| 빼기로 지워지나 | ✅ 지운 기억 정답률 0% = 정답지(처음부터 안 배움)와 같음. 3 seed | `runs/gpu/del_4b_r32.jsonl` |
| 남은 기억이 망가지나 | ⚠️ 조금 (정답률 6~14%p 하락, NLL +0.21). **무관한 사람(decoy)의 변화량을 빼도 똑같이 망가짐**, 무작위 방향은 영향 없음 → 손상 원인은 "기억 하나 배울 때 생기는 공통 변화" | 같은 파일, `sub_decoy`/`sub_random` 열 |
| 손상 줄이는 방법 | ✅ ΔB_k 중 decoy 방향 성분은 남기고 빼면 손상 절반 (+0.21 → +0.09). ❌ 다른 기억들 평균 방향은 효과 없음 | `runs/gpu/del_variants_4b_r32.jsonl` |
| 지운 뒤 뭐라고 답하나 | ⚠️ "모르겠다"가 아니라 다른 사람 답을 가져다 씀 (12개 중 10~12개) | `del_4b_r32.jsonl`의 `gen_target_direct.types` |
| 논문 재현 (LongMemEval 50문제, 4B) | ❌ 기억 없음 F1 8.4 (논문 5.3 ✅), TMEM F1 9.8 (논문 41.2). 학습 2배 세게 하면 5.2로 더 나빠짐 | `runs/gpu/lme_*.jsonl` |
| 재현 차이 원인 | ⚠️ 논문에 답변 프롬프트가 없음. 그림 설명은 "검색된 QA 쌍"을 글로 보여 준 것으로 읽힘. 뽑은 QA를 글로 보여 주니 F1 32~35 (진행 중, 22~24/50) | `docs/paper-vs-impl-20261105.md` |
| 선행 연구 | 빼기(Amnesiac ML), 공통 성분 분리(Ethos 등), decoy 비슷한 생각("Unlearning Can Hurt When There Is Nothing to Forget")은 있음. **"평균 방향 vs 같은 시점 decoy" 대비는 못 찾음** | `docs/related-work-ars-20261105.md`, `docs/novelty-check-20261105.md` |

## 3. 문제와 방향 전환 (사용자 결정, 2026-11-05)
- 실제로 일하는 에이전트는 큰 API 모델을 쓰고 가중치를 못 건드린다 → 가중치 기억 자체가 제품에 안 맞음.
- 재현에서도 점수 대부분이 글로 보여 준 QA에서 나옴 (확인 중).
- **새 방향: 글 기억 삭제.** 원본을 지워도 누적 요약·파생 메모·검색 색인에 새어 남는지, 어떤 방법이 싸고 정확하게 다 지우나. 설계: `docs/plan-text-memory-deletion.md`.

## 4. 지금 돌고 있는 것 (GPU 4)
| 작업 | 명령 | 결과 파일 |
|---|---|---|
| 기억 + QA 글로 보여 주기 | `run_longmemeval.py --answer-qa all` | `runs/lme_qatext_delta.jsonl` (노드) |
| QA 글만, 기억 끔 | `--answer-qa all --no-delta-at-answer` | `runs/lme_qatext_nodelta.jsonl` (노드) |
끝나면 (약 1.5시간): 노드 `~/tmem-repro/runs/`에서 `runs/gpu/`로 복사, 50문제 F1 비교. 둘이 비슷하고 논문 값 근처면 "논문 점수는 글에서 나온다"로 결론.

## 5. 이어서 할 일 (순서)
1. 4번 결과 정리 → 가중치 연구 사내 보고서 1장 (결과표 + decoy 발견 + 재현 결론). workshop 초안은 선택.
2. 글 기억 삭제 선행 연구 조사 (ARS `deep-research`: RAG unlearning, 요약 속 정보 누출, provenance 기반 삭제).
3. 글 기억 실험 코드: 기억 4종(원본/누적 요약/파생 메모/검색 색인) × 삭제 4종(원본만/출처 추적 재생성/LLM 검열/정답지). 합성 인물 데이터(`tmem/synth.py`)와 답 분류(`tmem/answers.py`) 재사용. 에이전트 모델은 Qwen3를 vLLM으로.
4. (선택) 가중치 쪽 미완: binary_ko(한국어 2지선다) 삭제 실험, 0.6B 비교, 복습이 다른 기억 변화량에 새는지.

## 6. 코드 지도
| 파일 | 역할 |
|---|---|
| `tmem/svd_lora.py` | SVD-init LoRA, FastWeights(state/load/add), 변화량 유틸 |
| `tmem/update.py` | online SFT (SGD) |
| `tmem/synth.py` | 합성 인물 스트림(open/binary_ko), 실제 단어 풀, 문장 답, decoy |
| `tmem/answers.py` | 생성 답 유형 분류 (gold/partial/self_other/other_memory/refuse/garbled/other) |
| `tmem/agent.py`, `tmem/prompts.py` | LongMemEval용 TMEM 에이전트. `answer_qa`, `delta_at_answer` 옵션 |
| `scripts/run_deletion.py` | 실험 1: full/sub/oracle + α/random/decoy/repeat 대조, 복습(`--rehearse`), ΔB 저장 |
| `scripts/run_deletion_variants.py` | 저장한 ΔB로 삭제 방식 4종 비교 |
| `scripts/run_retention.py` | 설정별 기억 유지 곡선 (복습 포함) |
| `scripts/run_longmemeval.py` | 논문 재현 |
| `scripts/analyze_deltas.py`, `summarize_deletion.py`, `brief.py` | 분석·요약·브리핑 |
| `docs/` | 선행 연구 2건, 논문 대조, 방향 전환 설계 |
테스트: `PYTHONPATH=. .venv/bin/python -m pytest -q` (60 passed). GPU 환경: `/mnt/sdb/cache/khope/envs/tmem/bin/python`, 코드는 rsync로 `~/tmem-repro/`. 저장한 ΔB: 노드 `/mnt/sdb/cache/khope/tmem_deltas/` (seed당 1GB, Mac에 없음).

## 7. 산출물
- 연구 요약 그림: `~/Downloads/imagen/tmem-research-brief_1.png` (논문 재현 결과 반영 전 버전)
- GPU 도우미: `~/.agents/skills/sionic-onprem-gpu/scripts/gpu.sh` (free/report/run/status/wait)
