"""
    GOAL: 
    Before any optimization, we need to have baseline to compare to.
    --> warm-up for GPU is non-negociable
    --> several runs are necessary to alleviate GPU/CPU flucutations
        and accurate statistical comparison.
"""


import os
import csv
import time
import statistics as stats
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset

from TarFlow.architecture import Model
from TarFlow.utils import set_random_seed


#### Used to fix the number of workers.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))


###################
# Baseline config #
################### 

torch.set_float32_matmul_precision("high")

RANDOM_SEED = 200
set_random_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("This baseline is to measure GPU processes. Please run it on GPU equipped computer.")



## CONFIG HYPERPARAMS
BATCH_SIZE = 256 # better to be a power of 2 to leverage GPU architecture
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1

FACTOR = 1.0
RESCALE_FACTOR = 1 / FACTOR
SIGMA_MAX = 0.0
DATA_PATH = "data/train_set_tensor.pt"

NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '0'))

IMG_SIZE = 64
IN_CHANNELS = 3
PATCH_SIZE = 2
CHANNELS = 128
NUM_BLOCKS = 1
LAYERS_PER_BLOCK = 8
NVP = True
NUM_CLASSES = 0

USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16

# Baseline protocol
WARMUP_STEPS = 10 # avoid costs linked to cuda init, libraries, etc in the first batches
MEASURE_STEPS = 50
REPEATS = 5

# False = to get a clean baseline (because item synchronize but for the cuda events we need to 
#         synchronize. I'm not sure how we'll play this right now. TOBEDISCUSSED.
MEASURE_ITEM_SYNC = False

RUN_NAME = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR = os.path.join("baselines", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "baselines_step_times.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "baselines_summary.txt")


########
# Data #
########

def build_dataloader(
    data_path: str,
    batch_size: int,
    sigma_max: float,
    num_workers: int,
    size_data: int | None = None,
) -> DataLoader:
    raw_data = torch.load(data_path, weights_only=True)

    if isinstance(raw_data, dict):
        data_train_x = raw_data.get("x")
        if data_train_x is None:
            raise ValueError("Data dictionary does not contain the key 'x'.")
    else:
        data_train_x = raw_data

    if size_data is not None:
        data_train_x = data_train_x[:size_data]

    data_train_x = data_train_x + sigma_max * torch.randn_like(data_train_x)

    dataset = TensorDataset(data_train_x)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


#########
# Model #
#########

def build_model_and_optimizer():
    model = Model(
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        num_blocks=NUM_BLOCKS,
        layers_per_block=LAYERS_PER_BLOCK,
        nvp=NVP,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    return model, optimizer



##################################################################################
# IMPORTANT: To measure anything, use cudaevent and cudasynchronize right after! #
##################################################################################

def cuda_event():
    return torch.cuda.Event(enable_timing=True)

def elapsed_ms(start_event, end_event) -> float:
    return start_event.elapsed_time(end_event)

def summarize(values):
    return {
        "mean": stats.mean(values),
        "median": stats.median(values),
        "min": min(values),
        "max": max(values),
        "p90": sorted(values)[int(0.90 * (len(values) - 1))], # 90th percentile
    }


def print_env():
    print("=" * 80)
    print("BASELINE ENV")
    print("=" * 80)
    print(f"device: {DEVICE}")
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_version_torch: {torch.version.cuda}")
    print(f"matmul_precision: high")
    print(f"amp_enabled: {USE_AMP}")
    print(f"amp_dtype: {AMP_DTYPE}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"num_workers: {NUM_WORKERS}")
    print(f"warmup_steps: {WARMUP_STEPS}")
    print(f"measure_steps: {MEASURE_STEPS}")
    print(f"repeats: {REPEATS}")
    print(f"measure_item_sync: {MEASURE_ITEM_SYNC}")
    print("=" * 80)


########################################################
# Measure of one training step (only GPU ops in there) #
########################################################

def run_one_step(model, optimizer, batch):
    timings = {}

    # measure mem transfers host to device 
    # + rescale
    h2d_start = cuda_event()
    h2d_end = cuda_event()

    h2d_start.record()

    if len(batch) == 2:
        x, y = batch
        y = y.to(DEVICE, non_blocking=True)
    else:
        (x,) = batch
        y = None

    # does non-blocking alone force non-pageable mem
    # or need pinned_mem to be True necessarily?
    x = x.to(DEVICE, non_blocking=True)
    x = x * RESCALE_FACTOR

    h2d_end.record()

    # measure forward + loss
    fwd_start = cuda_event()
    fwd_end = cuda_event()

    fwd_start.record()

    with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
        z, outputs, logdets = model(x, y)
        loss = model.get_loss(z, logdets)

    fwd_end.record()

    # measure backward only (problem with accumulation step != 1 no?)
    bwd_start = cuda_event()
    bwd_end = cuda_event()

    bwd_start.record()

    (loss / ACCUMULATION_STEPS).backward()

    bwd_end.record()

    # measure prior update
    prior_start = cuda_event()
    prior_end = cuda_event()

    prior_start.record()

    with torch.no_grad():
        model.update_prior(z)

    prior_end.record()

    # measure opt
    opt_start = cuda_event()
    opt_end = cuda_event()

    opt_start.record()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    opt_end.record()

    # False by default for now
    item_sync_ms = 0.0
    loss_value = None

    if MEASURE_ITEM_SYNC:
        item_t0 = time.perf_counter()
        # if we have item here, since it's already synchronizing
        # GPU-CPU, is the torch cuda synchronize necessary?
        loss_value = loss.detach().item()
        torch.cuda.synchronize()
        item_t1 = time.perf_counter()
        item_sync_ms = (item_t1 - item_t0) * 1000.0

    # one sync for everybody!
    torch.cuda.synchronize()

    # /!\ GPU measures are performed in ms
    timings["h2d_ms"] = elapsed_ms(h2d_start, h2d_end)
    timings["forward_loss_ms"] = elapsed_ms(fwd_start, fwd_end)
    timings["backward_ms"] = elapsed_ms(bwd_start, bwd_end)
    timings["update_prior_ms"] = elapsed_ms(prior_start, prior_end)
    timings["optimizer_ms"] = elapsed_ms(opt_start, opt_end)
    timings["item_sync_ms"] = item_sync_ms

    if loss_value is None:
        loss_value = loss.detach().float().cpu().item()

    timings["loss"] = float(loss_value)

    return timings


##########################################################################
# Main benchmark: we split CPU and GPU time measures                     #
# IMPORTANT: CPU time is in s but converted to ms to match GPU time unit #
##########################################################################

def main():
    print_env()

    train_loader = build_dataloader(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        sigma_max=SIGMA_MAX,
        num_workers=NUM_WORKERS,
    )

    print(f"dataset_size: {len(train_loader.dataset)}")
    print(f"num_batches_per_epoch: {len(train_loader)}")
    print(f"output_dir: {OUT_DIR}")

    all_rows = []

    for repeat in range(REPEATS):
        print(f"\n--- repeat {repeat + 1}/{REPEATS} ---")

        model, optimizer = build_model_and_optimizer()
        model.train()

        # IMPORTANT to reset mem! 
        # (if I'm correct, there must be no active tensors bc of the 2 previous lines.
        #  Can someone check to see whether he agrees or not? )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        loader_iter = iter(train_loader)

        # IMPORTANT to warmup before any measure
        # earlier batch realize more mem transfers and we care 
        # about the average batch to perform optim.
        total_steps = WARMUP_STEPS + MEASURE_STEPS

        for step_idx in range(total_steps):
            step_wall_t0 = time.perf_counter()

            # dataloader wait (CPU process)
            data_t0 = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration: 
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            data_t1 = time.perf_counter()

            # GPU measure
            timings = run_one_step(model, optimizer, batch)

            step_wall_t1 = time.perf_counter()

            is_warmup = step_idx < WARMUP_STEPS

            row = {
                "repeat": repeat,
                "step_idx": step_idx,
                "is_warmup": int(is_warmup),
                # dataloader
                "data_wait_ms": (data_t1 - data_t0) * 1000.0,
                # global time
                "step_wall_ms": (step_wall_t1 - step_wall_t0) * 1000.0,
                **timings,
                "batch_size": BATCH_SIZE,
                "samples_per_sec_gpu": BATCH_SIZE / (timings["h2d_ms"]
                                                 + timings["forward_loss_ms"]
                                                 + timings["backward_ms"]
                                                 + timings["update_prior_ms"]
                                                 + timings["optimizer_ms"]) * 1000.0,
                "samples_per_sec_wall": BATCH_SIZE / (step_wall_t1 - step_wall_t0),
                # conversion to gb
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            }

            all_rows.append(row)

            tag = "warmup" if is_warmup else "measure"
            print(
                f"[{tag}] repeat={repeat} step={step_idx} "
                f"wall={row['step_wall_ms']:.2f} ms | "
                f"data={row['data_wait_ms']:.2f} | "
                f"h2d={row['h2d_ms']:.2f} | "
                f"fwd={row['forward_loss_ms']:.2f} | "
                f"bwd={row['backward_ms']:.2f} | "
                f"opt={row['optimizer_ms']:.2f} | "
                f"loss={row['loss']:.4f}"
            )

        del model
        del optimizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Save CSV
    fieldnames = list(all_rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    measured = [r for r in all_rows if r["is_warmup"] == 0]

    metrics = [
        "step_wall_ms",
        "data_wait_ms",
        "h2d_ms",
        "forward_loss_ms",
        "backward_ms",
        "update_prior_ms",
        "optimizer_ms",
        "item_sync_ms",
        "samples_per_sec_gpu",
        "samples_per_sec_wall",
        "max_memory_allocated_gb",
        "max_memory_reserved_gb",
    ]

    lines = []
    lines.append("=" * 80)
    lines.append("BASELINE SUMMARY, warmup excluded")
    lines.append("=" * 80)

    for metric in metrics:
        values = [r[metric] for r in measured]
        s = summarize(values)
        lines.append(
            f"{metric:28s} "
            f"mean={s['mean']:.4f} "
            f"median={s['median']:.4f} "
            f"p90={s['p90']:.4f} "
            f"min={s['min']:.4f} "
            f"max={s['max']:.4f}"
        )

    first_loss = measured[0]["loss"]
    last_loss = measured[-1]["loss"]
    lines.append(f"loss_first_measured={first_loss:.6f}")
    lines.append(f"loss_last_measured={last_loss:.6f}")
    lines.append(f"csv_path={CSV_PATH}")

    summary = "\n".join(lines)

    print("\n" + summary)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
