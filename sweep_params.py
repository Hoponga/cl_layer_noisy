#!/usr/bin/env python
"""
Launch a hyper-parameter sweep where **multiple independent runs
share the same GPU concurrently**.

• Each worker picks     gpu_id = local_rank % N_GPUS
• It sets `torch.cuda.set_device(gpu_id)`          (no DDP)
• Calls mean_teacher.train_and_eval(cfg)
• Writes one CSV line with per-task + average accuracy.

You control *how many* workers per GPU with `WORKERS_PER_GPU`.
"""

import os, csv, itertools, datetime, random, socket
import multiprocessing as mp
import torch
from single_train_camel import train_and_eval          # your model file

# ---------------- hyper-parameter grid ------------------------
K_vals        = [10, 15, 20]
D_TSK_vals    = [32, 48, 64]

#  three orders of magnitude sweep: 10⁻³, 10⁻², 10⁻¹, 10⁰
lam_vals      = torch.logspace(-3, 0, steps=4).tolist()   # [0.001, 0.01, 0.1, 1.0]

lambda_tuples = list(itertools.product(lam_vals,
                                       lam_vals,
                                       lam_vals))        # 4³ = 64

grid = list(itertools.product(K_vals, D_TSK_vals, lambda_tuples))
random.shuffle(grid)           # avoid all heavy configs on same GPU
# --------------------------------------------------------------

WORKERS_PER_GPU = 8            # ← how many concurrent procs per GPU
N_GPUS          = torch.cuda.device_count()
TOTAL_WORKERS   = min(len(grid), N_GPUS * WORKERS_PER_GPU)


def worker(proc_idx, combos):
    """
    Simple independent worker:
      • choose GPU
      • run one training job
      • write results CSV
    """
    gpu_id = proc_idx % N_GPUS
    torch.cuda.set_device(gpu_id)

    if proc_idx >= len(combos):
        return

    K, D_TSK, (lam_o, lam_m, lam_wd) = combos[proc_idx]

    cfg = dict(
        K         = K,
        D_TSK     = D_TSK,
        LAM_ORTHO = lam_o,
        LAM_M     = lam_m,
        LAM_WD    = lam_wd,
        device    = f"cuda:{gpu_id}",
    )

    print(f"starting {cfg}")

    acc = train_and_eval(cfg)
    avg = sum(acc)/len(acc)

    ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"run_{proc_idx:03d}_gpu{gpu_id}_{ts}.csv"
    with open(filename, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["K","D_TSK","lam_o","lam_m","lam_wd"] +
                     [f"task{i}" for i in range(len(acc))] + ["avg"])
        wr.writerow([K, D_TSK, lam_o, lam_m, lam_wd,
                     *[f"{a:.4f}" for a in acc], f"{avg:.4f}"])
    print(f"[PID {os.getpid()}] finished {filename}")


if __name__ == "__main__":
    print(f"GPUs available: {N_GPUS}")
    print(f"Launching {TOTAL_WORKERS} workers "
          f"({WORKERS_PER_GPU} per GPU)…")

    # Spawn independent processes
    mp.set_start_method("spawn", force=True)
    procs = []
    for idx in range(TOTAL_WORKERS):
        p = mp.Process(target=worker, args=(idx, grid))
        p.start(); procs.append(p)

    # Wait for completion
    for p in procs:
        p.join()

    print("Sweep complete.")
