import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from TarFlow.architecture import Model
from TarFlow.utils import set_random_seed
#from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler




# ============================================================
## ⚙️ Initial Configuration and Hardware
# ============================================================
torch.set_float32_matmul_precision("high")

RANDOM_SEED = 200
set_random_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

NUM_WORKERS = 0

print(f"⚙️ Using FACTOR: {FACTOR}")
print(f"⚙️ Device: {DEVICE}")



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
## 💾 Data Setup
# ============================================================
def build_dataloader(data_path: str, batch_size: int, sigma_max: float,
                    num_workers: int, size_data: int | None = None) -> DataLoader:
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
    
    # Here what does pin_memory do ?  
    # num_workers how does it work ?
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader

train_loader = build_dataloader(
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    sigma_max=SIGMA_MAX,
    num_workers=NUM_WORKERS,
)

total_batches = len(train_loader)
print(f"✅ Data loaded: {len(train_loader.dataset)} samples, {total_batches} batches.")

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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=1e-4,
)

# Mixed precision (bf16) 
USE_AMP = DEVICE.type == "cuda"
amp_dtype = torch.bfloat16

print("✅ Data, Model, and Optimizer initialized.")

# ============================================================
## 📝 Logging and Checkpointing
# ============================================================
RUN_NAME = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

writer = SummaryWriter(log_dir=LOG_DIR)

SAVE_PATH = RUN_NAME
CKPT_FILE = os.path.join("flow_models", f"{SAVE_PATH}.ckpt")
print(f"💾 Checkpoint will be saved as: {CKPT_FILE}")

LOG_EVERY_N_STEPS = 10




# ============================================================
## Profiling parameters and functions
# ============================================================

# LOG_DIR_PROFILING = os.path.join("profiling_torch", RUN_NAME)
# os.makedirs(LOG_DIR_PROFILING, exist_ok=True)

# PROFILE_DIR = os.path.join(LOG_DIR_PROFILING, "profiler")
# os.makedirs(PROFILE_DIR, exist_ok=True)


# activities = [ProfilerActivity.CPU]
# if DEVICE.type == "cuda":
#     activities.append(ProfilerActivity.CUDA)

# prof = profile(
#     activities=activities, # I don't know what this is
#     record_shapes=True, # I don't know what this is
#     profile_memory=True, # I don't know what this is
#     with_stack=False, # I don't know what this is
#     on_trace_ready=tensorboard_trace_handler(PROFILE_DIR),
# )



# ============================================================
## 🚀 Training Loop
# ============================================================
print(f"🔥 Starting Training for {EPOCHS} epochs...")

global_step = 0

for epoch in range(EPOCHS):
    
    # if epoch==2:
    #     prof.start()
    model.train()

    epoch_loss_sum = 0.0
    epoch_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        
        
        # Unpack: dataset is TensorDataset(x) so batch is a tuple (x,)
        if len(batch) == 2:
            x, y = batch
            # Why non_blocking=True ?
            y = y.to(DEVICE, non_blocking=True)
        else:
            (x,) = batch
            y = None

        x = x.to(DEVICE, non_blocking=True)
        x = x * RESCALE_FACTOR

        with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=USE_AMP):
            z, outputs, logdets = model(x, y)
            loss = model.get_loss(z, logdets)

        (loss / ACCUMULATION_STEPS).backward()

        # Update prior (running variance) – done in fp32, no grad
        with torch.no_grad():
            model.update_prior(z)

        # Gradient step if we've accumulated enough
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        # What does detach here do exactly ? 
        loss_val = loss.detach().item()
        epoch_loss_sum += loss_val
        epoch_batches += 1
        global_step += 1

        if global_step % LOG_EVERY_N_STEPS == 0:
            writer.add_scalar("train/loss_step", loss_val, global_step)
            print(
                f"Epoch {epoch+1}/{EPOCHS} | step {global_step} | "
                f"batch {batch_idx+1}/{len(train_loader)} | loss {loss_val:.4f}"
            )

    # Flush remaining gradients if the last accumulation window wasn't complete
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Epoch-level metrics
    avg_loss = epoch_loss_sum / max(epoch_batches, 1)
    prior_var_mean = model.var.mean().item()
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    writer.add_scalar("train/prior_var_mean", prior_var_mean, epoch)
    print(
        f"Epoch {epoch+1} done | avg loss: {avg_loss:.4f} | "
        f"prior_var_mean: {prior_var_mean:.4f}"
    )

    # Checkpoint at the end of every epoch (overwrites previous one,
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
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
    # if epoch==2:
    #     prof.step()
    #     prof.stop()
    


writer.close()
print("\n✅ Training complete!")