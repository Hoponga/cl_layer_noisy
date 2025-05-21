import os, csv, itertools, datetime, json, torch, torch.multiprocessing as mp
from single_train_camel import train_and_eval        # <- you write this
import numpy as np

# ------------------------------------------------------------------
#  grid  -- change lambdas here if you want more/other values
# ------------------------------------------------------------------
K_vals        = [10, 15]
D_tsk_vals    = [32, 48]
lambda_tuples = [(0.05, .5, .5),
                 (0.10, 1., 1.),
                 (0.20, 2., 2.)]

lam_vals = np.logspace(-3, 0, num=4, base=10.0)    # [0.001, 0.01, 0.1, 1.0]

#  all 4×4×4 = 64 combinations
lambda_tuples = list(itertools.product(lam_vals, lam_vals, lam_vals))


grid = list(itertools.product(K_vals, D_tsk_vals, lambda_tuples))
print(grid)
# e.g.  3 * 3 * 3  = 27 jobs
# ------------------------------------------------------------------

def run_one(rank, world_size, combos):
    """
    Each DDP rank grabs one combo from `combos[rank]` and trains.
    """
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl",
        rank=rank, world_size=world_size)

    # --------------- pick the combo for this rank -----------------
    print(rank, len(combos))
    if rank >= len(combos):
        print(f"Rank {rank}: idle (fewer combos than GPUs)")
        return

    K, D_TSK, (lam_o, lam_m, lam_wd) = combos[rank]
    cfg = dict(
        K          = K,
        D_TSK      = D_TSK,
        LAM_ORTHO  = lam_o,
        LAM_M      = lam_m,
        LAM_WD     = lam_wd,
        device     = f"cuda:{rank}",
        rank       = rank,
    )

    # --------------- do the actual training -----------------------
    task_acc = train_and_eval(cfg)          # → list[10] of floats
    avg_acc  = sum(task_acc) / len(task_acc)

    # --------------- write CSV ------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"results_rank{rank}_{timestamp}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["K","D_TSK","lam_ortho","lam_M","lam_WD"] + \
                 [f"task{i}" for i in range(len(task_acc))] + ["avg"]
        writer.writerow(header)
        writer.writerow([K, D_TSK, lam_o, lam_m, lam_wd,
                         *[f"{a:.4f}" for a in task_acc],
                         f"{avg_acc:.4f}"])
    print(f"Rank {rank}: finished {cfg}  →  {fname}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # --------------------------------------------------------------------
    #  Decide how many worker processes to launch
    # --------------------------------------------------------------------
    n_gpu      = torch.cuda.device_count()
    n_jobs     = len(grid)
    n_workers  = min(n_gpu, n_jobs)     # launch only what we need

    if n_workers == 0:
        raise RuntimeError("No CUDA devices or empty job grid!")

    print(f"Launching {n_workers} workers for {n_jobs} hyper-configs "
          f"on {n_gpu} GPU(s)")

    mp.spawn(
        run_one,
        args=(n_workers, grid),   # pass the full grid; run_one will skip if rank>=len(grid)
        nprocs=n_workers,
        join=True
    )