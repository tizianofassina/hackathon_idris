import os
from datetime import datetime
import torch
from torch.cuda import nvtx
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from TarFlow.architecture import Model
from TarFlow.utils import set_random_seed

import torch.distributed as dist  # DDP communication
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import time

# ============================================================
## ⚙️ Initial Configuration and Hardware
# ============================================================
torch.set_float32_matmul_precision("high")
RANDOM_SEED = 200
set_random_seed(RANDOM_SEED)

# ============================================================
## 🛠️ Training Parameters and Hyperparameters
# ============================================================
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1

FACTOR = 1.0
RESCALE_FACTOR = 1 / FACTOR
SIGMA_MAX = 0.0
DATA_PATH = "data/train_set_tensor.pt"


# ============================================================
## 🏗️ Model Architecture Parameters
# ============================================================
IMG_SIZE = 64
IN_CHANNELS = 3
PATCH_SIZE = 2
CHANNELS = 128
NUM_BLOCKS = 1
LAYERS_PER_BLOCK = 8
NVP = True
NUM_CLASSES = 0


# ============================================================
## Setting up Distributed Data Parallel (DDP)
# ============================================================

def setup_ddp():
    """
    Initialize the distributed process group.
    Reads RANK / LOCAL_RANK / WORLD_SIZE from environment variables
    (set automatically by torchrun).
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# ============================================================
## 💾 Data Setup
# ============================================================
def build_dataloader(data_path: str, batch_size: int, sigma_max: float,
                    num_workers: int, size_data: int | None = None):
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

    # DistributedSampler splits the dataset across processes
    sampler = DistributedSampler(dataset, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,                  # shuffle handled by sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, sampler


def main(local_rank):
    # ============================================================
    ## Device + rank info
    # ============================================================
    DEVICE = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    is_main = (rank == 0)

    if is_main:
        print(f"⚙️ Using FACTOR: {FACTOR}")
        print(f"⚙️ World size: {dist.get_world_size()}")

    os.makedirs("flow_models", exist_ok=True)

    # ============================================================
    ## 🏗️ Model Setup
    # ============================================================
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

    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    # Mixed precision (bf16)
    amp_dtype = torch.bfloat16

    if is_main:
        print("✅ Data, Model, and Optimizer initialized.")

    # ============================================================
    ## 📝 Logging and Checkpointing (rank 0 only)
    # ============================================================
    writer = None
    CKPT_FILE = None
    if is_main:
        RUN_NAME = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        LOG_DIR = os.path.join("runs", RUN_NAME)
        os.makedirs(LOG_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=LOG_DIR)
        CKPT_FILE = os.path.join("flow_models", f"{RUN_NAME}.ckpt")
        print(f"💾 Checkpoint will be saved as: {CKPT_FILE}")

    LOG_EVERY_N_STEPS = 10

    fp16_scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ============================================================
    ## DataLoader
    # ============================================================
    # Read CPUs per task from SLURM if available, fallback to 8
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    num_workers = min(num_workers, 8)

    train_loader, train_sampler = build_dataloader(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        sigma_max=SIGMA_MAX,
        num_workers=num_workers,
        size_data=None,
    )

    total_batches = len(train_loader)
    if is_main:
        print(f"✅ Data loaded: {len(train_loader.dataset)} samples, "
            f"{total_batches} batches per process.")
        print(f"🔥 Starting Training for {EPOCHS} epochs...")

    # ============================================================
    ## 🚀 Training Loop
    # ============================================================
    global_step = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        # Start CUDA profiler on epoch 2 (all ranks profile, nsys captures each)
        if epoch == 2:
            torch.cuda.cudart().cudaProfilerStart()

        model.train()

        epoch_loss_sum = 0.0
        epoch_batches = 0

        nvtx.range_push("Dataloader")
        for batch_idx, batch in enumerate(train_loader):
            nvtx.range_pop()

            nvtx.range_push("Copying to Device")
            if len(batch) == 2:
                x, y = batch
                y = y.to(DEVICE, non_blocking=True)
            else:
                (x,) = batch
                y = None
            x = x.to(DEVICE, non_blocking=True)
            x = x * RESCALE_FACTOR
            nvtx.range_pop()

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                nvtx.range_push("Forward pass")
                z, outputs, logdets = model(x, y)
                loss = model.module.get_loss(z, logdets)
                nvtx.range_pop()

            nvtx.range_push("Backward pass")
            fp16_scaler.scale(loss / ACCUMULATION_STEPS).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            nvtx.range_pop()
            optimizer.zero_grad(set_to_none=True)

            # Update prior (running variance) – done in fp32, no grad
            with torch.no_grad():
                model.module.update_prior(z)

            # Logging
            nvtx.range_push("Logging loss")
            loss_val = loss.detach().item()
            epoch_loss_sum += loss_val
            epoch_batches += 1
            global_step += 1

            if is_main and global_step % LOG_EVERY_N_STEPS == 0:
                writer.add_scalar("train/loss_step", loss_val, global_step)
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | step {global_step} | "
                    f"batch {batch_idx+1}/{total_batches} | loss {loss_val:.4f}"
                )
            nvtx.range_pop()

            nvtx.range_push("Dataloader")

        # End of epoch
        # Aggregate loss across all processes
        avg_loss_tensor = torch.tensor(
            [epoch_loss_sum / max(epoch_batches, 1)], device=DEVICE
        )
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        prior_var_mean = model.module.var.mean().item()

        if is_main:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            writer.add_scalar("train/prior_var_mean", prior_var_mean, epoch)
            print(
                f"Epoch {epoch+1} done | avg loss: {avg_loss:.4f} | "
                f"prior_var_mean: {prior_var_mean:.4f}"
            )

        if epoch == 2:
            torch.cuda.cudart().cudaProfilerStop()

    if is_main:
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hparams": {
                    "in_channels": IN_CHANNELS,
                    "img_size": IMG_SIZE,
                    "patch_size": PATCH_SIZE,
                    "channels": CHANNELS,
                    "num_blocks": NUM_BLOCKS,
                    "layers_per_block": LAYERS_PER_BLOCK,
                    "nvp": NVP,
                    "num_classes": NUM_CLASSES,
                    "lr": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "rescale_factor": RESCALE_FACTOR,
                    "sigma_max": SIGMA_MAX,
                },
            },
            CKPT_FILE,
        )

    if is_main:
        writer.close()
        print("\n✅ Training complete!")


if __name__ == "__main__":
    local_rank = setup_ddp()
    try:
        start_time = time.time()
        main(local_rank)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if dist.get_rank() == 0:
            print(f"Total training time: {elapsed_time:.2f} seconds")
    finally:
        cleanup_ddp()