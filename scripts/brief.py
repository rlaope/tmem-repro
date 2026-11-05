"""실행 중인 GPU 실험 상태를 한국어 표로 출력한다 (Mac에서 실행, 노드는 ssh 한 번으로 조회).

각 작업: 살아 있는지, 진행 x/y, 경과·남은 시간, 지금까지의 수치(진행 중 표시), 이상 신호.
이상 신호: 프로세스가 끝났는데 결과가 덜 찼음, GPU 유휴, 로그에 Error/Traceback/OOM/nan.
--changed-only: 지난 실행과 진행·상태·이상이 같으면 아무것도 출력하지 않는다 (loop용).
종료 코드: 0 정상, 2 이상 신호 있음.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HOST = "worker-6000pro-2"
GPU = 4
RUNS = "~/tmem-repro/runs"
STATE = os.path.expanduser("~/.cache/tmem-brief-state.json")

# name, pgrep pattern (regex, no parens), result file, expected result rows, log, metric kind
JOBS = [
    ("논문 재현 (기억 있음, 50문제)", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_longmemeval.py.*strategy tmem.*lme_tmem_4b_50\\.jsonl", "lme_tmem_4b_50.jsonl", 50, "lme.log", "lme"),
    ("논문 재현 (기억 없음, 50문제)", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_longmemeval.py.*strategy none", "lme_none_4b_50.jsonl", 50, "lme.log", "lme"),
    ("삭제 실험 4B + 복습 32 (3 seed)", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_deletion.py.*del4b_r32", "del_4b_r32.jsonl", 24, "del_4b_r32.log", "del"),
    ("논문 재현 새 학습 설정 (lr 1e-3, 10 epochs, 8층)", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_longmemeval.py.*lme_tmem_4b_50_v2", "lme_tmem_4b_50_v2.jsonl", 50, "lme_v2.log", "lme"),
    ("논문 재현: 기억 + QA 글로 보여 주기", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_longmemeval.py.*lme_qatext_delta", "lme_qatext_delta.jsonl", 50, "lme_qatext_delta.log", "lme"),
    ("논문 재현: QA 글만 (기억 끔)", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_longmemeval.py.*lme_qatext_nodelta", "lme_qatext_nodelta.jsonl", 50, "lme_qatext_nodelta.log", "lme"),
    ("삭제 방식 비교 (공통 변화 남기기)", "^/mnt/sdb/cache/khope/envs/tmem/bin/python scripts/run_deletion_variants.py", "del_variants_4b_r32.jsonl", 15, "del_variants.log", "var"),
]

REMOTE = r"""
cd {runs} 2>/dev/null || exit 3
echo "@@GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits -i {gpu}
for j in {jobs}; do
  IFS='|' read -r pat file log <<< "$j"
  echo "@@JOB $file"
  pid=$(pgrep -f -- "$pat" | grep -vx $$ | head -1)
  if [ -n "$pid" ]; then echo "alive $(ps -o etimes= -p $pid | tr -d ' ')"; else echo "dead 0"; fi
  if [ -f "$file" ]; then echo "rows $(grep -c '"m"\|"qid"' "$file")"; tail -n 400 "$file" | grep -v '"config"' | tail -n 60; else echo "rows 0"; fi
  echo "@@LOG"; [ -f "$log" ] && grep -v "keep_after sub/orc=nan/nan" "$log" | grep -ciE "traceback|error:|out of memory|[ =]nan" || echo 0
done
"""


def fetch():
    jobs = " ".join("'%s|%s|%s'" % (p, f, l) for _, p, f, _, l, _ in JOBS)
    cmd = REMOTE.format(runs=RUNS, gpu=GPU, jobs=jobs)
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", HOST, "bash -s"], input=cmd,
                         capture_output=True, text=True, timeout=60)
    if out.returncode not in (0,) and "@@GPU" not in out.stdout:
        raise SystemExit(f"노드 조회 실패: {out.stderr.strip()[:200]}")
    return out.stdout


def hm(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}시간 {sec % 3600 // 60}분" if sec >= 3600 else f"{sec // 60}분"


def parse(raw: str):
    gpu, jobs, cur = None, {}, None
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln == "@@GPU":
            u, mu, mt, t, p = [x.strip() for x in lines[i + 1].split(",")]
            gpu = dict(util=float(u), mem=float(mu) / 1024, mem_total=float(mt) / 1024, temp=float(t), power=float(p))
            i += 2
            continue
        if ln.startswith("@@JOB "):
            cur = ln[6:]
            jobs[cur] = {"rows_data": []}
            st, sec = lines[i + 1].split()
            jobs[cur]["alive"], jobs[cur]["elapsed"] = st == "alive", float(sec)
            jobs[cur]["rows"] = int(lines[i + 2].split()[1])
            i += 3
            continue
        if ln == "@@LOG":
            jobs[cur]["log_errors"] = int(lines[i + 1].strip() or 0)
            i += 2
            continue
        if cur and ln.startswith("{"):
            try:
                jobs[cur]["rows_data"].append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        i += 1
    return gpu, jobs


def metric(kind: str, rows) -> str:
    if not rows:
        return "-"
    if kind == "lme":
        f1 = sum(r.get("f1", 0) for r in rows) / len(rows)
        return f"F1 {f1 * 100:.1f} (최근 {len(rows)}문제 평균, 진행 중)"
    r = rows[-1]
    if kind == "var":
        f = lambda v: f"{r[v]['acc_keep_before']:.0%}"  # noqa
        return (f"seed {r['seed']} m={r['m']} 앞 기억 정답률: 정답지 {f('oracle')}, 전체 빼기 {f('sub')}, "
                f"decoy 빼고 빼기 {f('sub_minus_decoy')}, 평균 방향 남기기 {f('sub_proj_mean')} | 지운 기억: "
                f"{r['sub_minus_decoy']['acc_target_direct']:.0%}/{r['sub_proj_mean']['acc_target_direct']:.0%}")
    s = r.get("sub", {}).get("gen_target_direct", {}).get("acc")
    o = r.get("oracle", {}).get("gen_target_direct", {}).get("acc")
    f = r.get("full", {}).get("gen_target_direct", {}).get("acc")
    kb = r.get("sub", {}).get("nll_keep_before", 0) - r.get("oracle", {}).get("nll_keep_before", 0)
    acc = f" | 지운 기억 정답률 학습/빼기/정답지 {f:.0%}/{s:.0%}/{o:.0%}" if s is not None else ""
    return (f"seed {r['seed']} m={r['m']}: 잔차 {r['residual_rel']:.2f}, 남은 기억 손상(NLL 차) {kb:+.2f}{acc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed-only", action="store_true")
    args = ap.parse_args()
    gpu, jobs = parse(fetch())
    alerts, rows_out, sig = [], [], {}
    for name, _, f, total, _, kind in JOBS:
        j = jobs.get(f, {"alive": False, "rows": 0, "elapsed": 0, "log_errors": 0, "rows_data": []})
        done = min(j["rows"], total)
        if j["alive"]:
            state = "✅ 실행 중"
        elif done >= total:
            state = "✅ 완료"
        elif done == 0:
            state = "⏳ 대기"
        else:
            state = "❌ 멈춤"
            alerts.append(f"{name}: 결과 {done}/{total}에서 프로세스가 없음")
        eta = hm(j["elapsed"] / done * (total - done)) if j["alive"] and done else ("❓" if j["alive"] else "-")
        if j.get("log_errors"):
            alerts.append(f"{name}: 로그에 error/traceback/OOM/nan {j['log_errors']}건")
        rows_out.append((name, state, f"{done}/{total}", hm(j["elapsed"]) if j["alive"] else "-", eta,
                         metric(kind, j["rows_data"])))
        sig[f] = [state, done, j.get("log_errors", 0)]
    any_alive = any(j.get("alive") for j in jobs.values())
    if gpu and any_alive and gpu["util"] < 5:
        alerts.append(f"GPU {GPU}번 사용률 {gpu['util']:.0f}% (작업이 있는데 놀고 있음)")
    if gpu and gpu["mem"] > 0.95 * gpu["mem_total"]:
        alerts.append(f"GPU {GPU}번 메모리 {gpu['mem']:.0f}/{gpu['mem_total']:.0f}GB (부족 위험)")
    sig["alerts"] = sorted(alerts)

    prev = None
    try:
        prev = json.load(open(STATE))
    except (OSError, json.JSONDecodeError):
        pass
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(sig, open(STATE, "w"))
    if args.changed_only and prev == sig:
        return

    print("| 작업 | 상태 | 진행 | 경과 | 남은 시간 | 지금까지 수치 |")
    print("|---|---|---|---|---|---|")
    for r in rows_out:
        print("| " + " | ".join(r) + " |")
    if gpu:
        print(f"\nGPU {GPU}번: 사용률 {gpu['util']:.0f}%, 메모리 {gpu['mem']:.0f}/{gpu['mem_total']:.0f}GB, "
              f"{gpu['temp']:.0f}°C, {gpu['power']:.0f}W")
    print("\n이상 신호: " + ("; ".join(alerts) if alerts else "없음"))
    sys.exit(2 if alerts else 0)


if __name__ == "__main__":
    main()
