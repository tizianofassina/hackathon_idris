#!/usr/bin/env python3
"""
GOAL: profile only few training steps after the baseline 
      has shown that fwd/bwd are the bottleneck.

Outputs:
    profiles/<run_name>/trace_step_<N>.json
    profiles/<run_name>/key_averages_cuda.txt
    profiles/<run_name>/key_averages_cuda_by_shape.txt
    profiles/<run_name>/key_averages_cpu.txt

Run with BATCH_SIZE=256 and NUM_WORKERS=0
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader, TensorDataset

from TarFlow.architecture import Model
from TarFlow.utils import set_random_seed


# Not the bottlenecks, 0, 256 --> OK
NUM_WORKERS = 0
BATCH_SIZE = 256 # better to be a power of 2 to leverage GPU architecture



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
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1

FACTOR = 1.0
RESCALE_FACTOR = 1 / FACTOR
SIGMA_MAX = 0.0
DATA_PATH = "data/train_set_tensor.pt"

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

# Profiler
PROFILE_WAIT   = 5
PROFILE_WARMUP = 5
PROFILE_ACTIVE = 10
PROFILE_REPEAT = 3
TOTAL_STEPS = (PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE) * PROFILE_REPEAT

# Profiler options
# show tensor shapes in key_averages_by_shape
RECORD_SHAPES = True
# track tensor allocations
PROFILE_MEMORY = True
# recommended to be False, don't know why TODO
WITH_STACK = False

## output
RUN_NAME = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
PROFILE_DIR = Path("profiles") / RUN_NAME
PROFILE_DIR.mkdir(parents=True, exist_ok=True)



def build_dataloader(
    data_path: str,
    batch_size: int,
    sigma_max: float,
    num_workers: int,
) -> DataLoader:
    raw_data = torch.load(data_path, weights_only=True)

    if isinstance(raw_data, dict):
        data_train_x = raw_data.get("x")
        if data_train_x is None:
            raise ValueError("Data dictionary does not contain key 'x'.")
    else:
        data_train_x = raw_data

    if sigma_max:
        data_train_x = data_train_x + sigma_max * torch.randn_like(data_train_x)

    dataset = TensorDataset(data_train_x)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )


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


def train_one_step(model, optimizer, batch, step_idx: int) -> float:
    """
        Return nothing and omit loss.time() to avoid parasitic 
        CPU-GPU synchronisation
    """
    with record_function("00_data_h2d"):
        if len(batch) == 2:
            x, y = batch
            y = y.to(DEVICE, non_blocking=True)
        else:
            (x,) = batch
            y = None

        x = x.to(DEVICE, non_blocking=True)
        x = x * RESCALE_FACTOR

    with record_function("01_forward_loss"):
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            z, outputs, logdets = model(x, y)
            loss = model.get_loss(z, logdets)

    with record_function("02_backward"):
        (loss / ACCUMULATION_STEPS).backward()

    with record_function("03_update_prior"):
        with torch.no_grad():
            model.update_prior(z)

    with record_function("04_optimizer_step"):
        if (step_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def main() -> None:
    print("=" * 80)
    print("TORCH PROFILER RUN")
    print("=" * 80)
    print(f"profile_dir: {PROFILE_DIR}")
    print(f"device: {DEVICE}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"cuda torch build: {torch.version.cuda}")
    print(f"data_path: {DATA_PATH}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"num_workers: {NUM_WORKERS}")
    print(f"img_size: {IMG_SIZE}")
    print(f"patch_size: {PATCH_SIZE}")
    print(f"channels: {CHANNELS}")
    print(f"num_blocks: {NUM_BLOCKS}")
    print(f"layers_per_block: {LAYERS_PER_BLOCK}")
    print(f"schedule: wait={PROFILE_WAIT}, warmup={PROFILE_WARMUP}, active={PROFILE_ACTIVE}, repeat={PROFILE_REPEAT}")
    print(f"record_shapes: {RECORD_SHAPES}")
    print(f"profile_memory: {PROFILE_MEMORY}")
    print(f"with_stack: {WITH_STACK}")
    print("=" * 80)

    train_loader = build_dataloader(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        sigma_max=SIGMA_MAX,
        num_workers=NUM_WORKERS,
    )

    model, optimizer = build_model_and_optimizer()
    model.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    def trace_handler(prof):
        step = prof.step_num
        trace_path = PROFILE_DIR / f"trace_step_{step}.json"
        cuda_table_path = PROFILE_DIR / "key_averages_cuda.txt"
        cuda_shape_table_path = PROFILE_DIR / "key_averages_cuda_by_shape.txt"
        cpu_table_path = PROFILE_DIR / "key_averages_cpu.txt"

        prof.export_chrome_trace(str(trace_path))

        cuda_table = prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=40,
        )
        cpu_table = prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=40,
        )

        cuda_table_path.write_text(cuda_table)
        cpu_table_path.write_text(cpu_table)

        if RECORD_SHAPES:
            cuda_shape_table = prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total",
                row_limit=60,
            )
            cuda_shape_table_path.write_text(cuda_shape_table)

        print("\n" + "=" * 80)
        print(f"TRACE READY: {trace_path}")
        print("=" * 80)
        print(cuda_table)

    prof_schedule = schedule(
        wait=PROFILE_WAIT,
        warmup=PROFILE_WARMUP,
        active=PROFILE_ACTIVE,
        repeat=PROFILE_REPEAT,
    )

    loader_iter = iter(train_loader)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=RECORD_SHAPES,
        profile_memory=PROFILE_MEMORY,
        with_stack=WITH_STACK,
    ) as prof:
        for step_idx in range(TOTAL_STEPS):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            train_one_step(model, optimizer, batch, step_idx)

            prof.step()

            print(f"step={step_idx + 1}/{TOTAL_STEPS}")

    torch.cuda.synchronize()

    print("\nDone.")
    print(f"Profile files written to: {PROFILE_DIR}")
    print("Open the trace JSON in https://ui.perfetto.dev")
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    print(f"Peak memory reserved:   {torch.cuda.max_memory_reserved() / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
