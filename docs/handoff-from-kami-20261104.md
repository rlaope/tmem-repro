# Handoff: revocable parametric memory research (from Kami, 2026-11-04)

모든 답은 한국어, 결론 먼저, 기술 용어는 영어 원어. 결과마다 ✅❌⚠️❓로 좋음/나쁨/조건부/모름 표시. 확인 안 한 것은 확인 안 했다고 쓴다.

## 연구 주제 (확정)
LLM 에이전트가 일하면서 공유 LoRA(fast weight)에 섞어 쌓은 기억 하나를, 요청 시 정확하고 싸게 지울 수 있는가.
가제: Revocable Persistent Parametric Memory for LLM Agents.
- 기반 논문: TMEM (arXiv 2606.04536). SVD-init LoRA, A 고정·B만 학습, plain SGD lr 5e-4, 5 epochs, batch 16, rank 6, 마지막 4층 FFN(gate/up/down). 공개 코드 없음 → 재구현.
- 참고: survey 2610.08630, SMF 2510.15103 (망각은 "배운 양 vs 잃은 양" 쌍으로 잰다).
- 되감기(그 기억 빼고 재학습)는 방법이 아니라 정답지(oracle). 실험 대상은 싼 삭제 방법 4개:
  1. 빼기: 쓰기마다 ΔB_k 저장, 지울 때 차감 (비용이 기억 나이와 무관)
  2. 잠자기 재정리: QA 장부에서 줄 삭제 후 주기적 재학습, 그 전까지 억누르기
  3. LoRA K칸 분할: 해당 칸만 재학습
  4. 직교 쓰기: 이전 기억 방향과 겹치지 않게 업데이트
- 평가 지표: 지운 사실을 아직 말하나(직접+우회 질문), 남은 기억·일반 능력 유지, 재학습 속도(정답지보다 빨리 돌아오면 억누른 것).
- 범위 밖: RL로 θ₀ 학습, memory layer, poisoning.
- 관련 연구 정리: ~/.hermes/profiles/kami/workspace/research/pmem-forget/related-work-20261030.md

## 코드
- repo: ~/work/tmem-repro (github rlaope/tmem-repro, private). git user rlaope. HEAD abc7247. `gh auth switch -u rlaope`.
- 로컬 venv: .venv (uv, py3.11). 테스트: `PYTHONPATH=. .venv/bin/python -m pytest -q` (9 passed).
- tmem/svd_lora.py: SVDLoRALinear, FastWeights(state/load/reset/add), state_diff/state_dist/state_norm
- tmem/update.py: online_sft (TMEM 업데이트 연산자)
- tmem/synth.py: 합성 사실 스트림 (가상 인물 1명 = 쓰기 1번 = 사실 F개, 학습 질문 2개 + 평가용 paraphrase 1개), GENERAL 상식 QA 20개
- tmem/agent.py, data.py, scripts/run_longmemeval.py: LongMemEval-S TMEM 재현 루프 (data/longmemeval_s_cleaned.json, gitignore)
- scripts/run_deletion.py: 실험 1 "빼기 흔적 vs k 뒤 쓰기 수 m". full / sub(=full−ΔB_k) / oracle(k 빼고 재학습) 비교, m 체크포인트, --gen-eval로 정확도, relearn 테스트.

## 지금까지 결과 (예비, seed 1개, 기억 9개, Qwen3-0.6B bf16, NLL만)
- ✅ m=0: sub == oracle (residual 0)
- ❌ m≥1: residual_rel 0.52(m=1) / 0.78(m=4) / 0.86(m=8)
- ⚠️ 지운 사실이 남는 것보다, 남겨야 할 기억이 더 망가짐: keep NLL sub 4.67 vs oracle 3.37 (m=1), 2.92 vs 2.03 (m=4)
- ❓ 정확도·seed 3개 미확인. 결론 아님.
- 결과 파일: worker-6000pro-2:~/tmem-repro/runs/del_quick_0.6b.{jsonl,log}

## GPU (중요 규칙)
- **khope가 명시적으로 허락하기 전엔 공용 GPU에서 아무것도 실행하지 않는다.** 접속·환경 준비는 실행 허락이 아니다.
- **실행 전 Creep Colony(https://live-creep-colony.sionic.im) GPU Status → 노드 → EDIT USAGE로 사용 신고.** SSH 실행은 자동 표시 안 됨. 끝나면 신고 삭제.
- 접속: `ssh worker-6000pro-1|worker-6000pro-2|b300-01..08` (user khope, key ~/.ssh/b300/id_ed25519, ~/.ssh/config 설정됨, NetBird VPN 필요). 공통 스킬: ~/.claude/skills/sionic-onprem-gpu/SKILL.md 먼저 읽기.
- 6000pro-2 준비물: 코드 ~/tmem-repro (rsync 복사본), venv /mnt/sdb/cache/khope/envs/tmem (torch 2.15 cu132, transformers 5.22.1).
- 쓰기 전 `nvidia-smi`로 빈 카드 확인, `CUDA_VISIBLE_DEVICES` 고정, 긴 작업은 nohup + 로그 파일.
- DGX 10.1.122.108: 이 계정/키로 접속 불가, 별도 계정 필요(khope가 비밀번호 확인 중).

## 다음 할 일 (khope 승인 후)
1. 실험 1 본 실행 설계 확정: Qwen3-0.6B, 기억 65개, m ∈ {0,1,2,4,8,16,32,64}, seed 3, --gen-eval. 이후 Qwen3-4B.
2. "keep이 망가진다"는 관찰 확인: 원인 가설(뒤 쓰기가 k 위에서 학습됨)을 ΔB 내적/방향 분석으로 검증.
3. 방법 2·3·4 구현 후 같은 지표로 비교.

## 진행 (2026-11-05, Claude Code)
- commit 491f55a..9340bf1 (push 완료): 답 중복 제거, keep 층별 채점, k=16 중간 삭제, α/random/decoy/repeat 대조, 답 유형 분류, 한국어 2지선다 사실(binary_ko), delta 분석·요약 스크립트, 기억 유지 확인 스크립트(run_retention.py).
- ❌ 논문 설정으로는 0.6B가 사실을 못 외우고 상식이 무너짐 (Mac/GPU 모두). 예비 결과의 "keep이 망가짐"은 원래 안 남은 기억으로 잰 것.
- ✅ 새 기본값(실제 단어, 문장 답, 질문 4표현, 일반 QA 8개 섞기, lr 1e-3, 10 epochs, 마지막 8층): 최신 기억 95~100%, 상식 유지.
- ❌ 0.6B는 새 설정에서도 기억 2개 이상 지나면 0%.
- 실행 중: worker-6000pro-2 GPU 4, run_retention.py 0.6B → 4B (runs/retention_{0.6b,4b}.jsonl, runs/retention.log). 끝나면 Creep Colony 신고 삭제.
- 다음: 4B가 4~7개 전 기억을 50% 이상 유지하면 4B로 실험 1 본 실행 설계 갱신.
