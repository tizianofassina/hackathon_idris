import torch
from TarFlowFFHQ.architecture import TarFlowModule, Model, TarFlowFFHQDataModule
from TarFlowFFHQ.utils import set_random_seed
from datetime import datetime
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os

# ============================================================
## ⚙️ Initial Configuration and Hardware
# ============================================================
# Optimization
torch.set_float32_matmul_precision("high")

# Set the seed for reproducibility
RANDOM_SEED = 200
set_random_seed(RANDOM_SEED)

# ============================================================
## 🛠️ Training Parameters and Hyperparameters
# ============================================================
# Core Training Parameters
BATCH_SIZE = 500  # Batch size for training
EPOCHS = 700 #Total number of epochs to train
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1  # Gradient accumulation steps
ENABLE_AMP = True  # Automatic Mixed Precision (AMP)

# Data/Scaling Parameters
FACTOR = 14  # 13  # 12, 11, 10 - Scaling factor for normalization
RESCALE_FACTOR = 1 / FACTOR  # Rescaling factor for the final output
SIGMA_MAX = 7  # 5.4710636138916015625 - Sigma parameter for noising
DATA_PATH = "datasets/train_set_tensor.pt"

print(f"⚙️ Using FACTOR: {FACTOR}")

# ============================================================
## 🏗️ Model Architecture Parameters
# ============================================================
IMG_SIZE = 64
IN_CHANNELS = 3
PATCH_SIZE = 2
# Modello grande dinamico (come da nome del checkpoint)
CHANNELS = 128  # 512,  # 128,  # 256,  # 128,  # 64,  # 128,  # 256  # 512, 768
NUM_BLOCKS = 1  # 1,  # 2,  # 2,  # 1,  # 2,  # 3  # 4, 8
LAYERS_PER_BLOCK = 8
NVP = True  # Non-Volume Preserving Flow
NUM_CLASSES = 0  # UNCONDITIONAL Training (e.g., FFHQ)

# ============================================================
## 💾 Data and Model Setup
# ============================================================

# 1️⃣ Initialize the Data Module
data_module = TarFlowFFHQDataModule(
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    sigma_max=SIGMA_MAX,
)
os.makedirs("flow_models", exist_ok=True)

# 2️⃣ Initialize the Core Model
model = Model(
    in_channels=IN_CHANNELS,
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    channels=CHANNELS,
    num_blocks=NUM_BLOCKS,
    layers_per_block=LAYERS_PER_BLOCK,
    nvp=NVP,
    num_classes=NUM_CLASSES,
)

# 3️⃣ Initialize the Lightning Module (Training Wrapper)
# The TarFlowModule defines all aspects within an epoch
tarflow_module = TarFlowModule(
    model=model,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    accum_steps=ACCUMULATION_STEPS,
    rescale_factor=RESCALE_FACTOR,
    enable_amp=ENABLE_AMP,
    sigma_max=SIGMA_MAX,
)

print("✅ Data, Model, and Lightning Module initialized.")

# ============================================================
## 📝 Logging and Checkpointing
# ============================================================
# Dynamic run name for logging


# Run name
RUN_NAME = f"FFHQ_noised_factor_{FACTOR}_SOTA_update_prior_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = os.path.join("runs", RUN_NAME)

# TensorBoard logger setup
# TensorBoardLogger is specific to Lightning . See https://lightning.ai/docs/pytorch/stable/extensions/logging.html for loggers
logger = TensorBoardLogger(save_dir="runs", name=RUN_NAME)

# Checkpoint file name
SAVE_PATH = f"TarFlow_FFHQ_noised_with_factor_{FACTOR}_SOTA_update_prior"

# For saving models during training. See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html for more informations
checkpoint_callback = ModelCheckpoint(
    dirpath="flow_models",
    filename=SAVE_PATH,
    save_top_k=1,
    save_last=False,
    monitor=None,
)
print(f"💾 Checkpoint will be saved as: {SAVE_PATH}.ckpt")
# print(save_path)

# ============================================================
## 🚀 Trainer Initialization and Training
# ============================================================

# Initialize the Lightning Trainer
# See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html for more informations
trainer = L.Trainer(
    accelerator="gpu",
    devices=torch.cuda.device_count(),
    precision="bf16-mixed",  # Precision setting for AMP
    max_epochs=EPOCHS,
    accumulate_grad_batches=ACCUMULATION_STEPS,
    default_root_dir=LOG_DIR,
    log_every_n_steps=10,
    logger=logger,
    callbacks=[checkpoint_callback],
    # resume_from_checkpoint=ckpt_path
)

# ===== Train =====
# Add after checkpoint_callback
# ckpt_path = f"flow_models/TarFlow_FFHQ_noised_with_factor_{FACTOR}_big_512_plateau.ckpt"
print(f"🔥 Starting Training for {EPOCHS} epochs...")
trainer.fit(tarflow_module, datamodule=data_module)  # , ckpt_path=ckpt_path)
print("\n✅ Training complete!")
