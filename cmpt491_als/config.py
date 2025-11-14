from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import torch

#config.py
# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# SAND-specific paths
SAND_DATA_DIR = RAW_DATA_DIR / "SAND" / "task1"
SAND_INTERIM_DIR = INTERIM_DATA_DIR / "SAND" / "task1"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Audio Processing Settings - things you might want to experiment with
AUDIO_CONFIG = {
    "target_sample_rate": 16000,    # AST Model was trained on 16khz clips, we have 8khz
    "max_length_seconds": 10.0,     # AST Model was trained on 10 sec clips
}

TRAINING_CONFIG_M2_MACBOOK = {
    "batch_size": 2,                # Small batch for M2 memory constraints
    "num_epochs": 10,
    "learning_rate": 1e-5,
    "num_workers": 2,               # Lower for macOS stability
    "gradient_accumulation_steps": 4, # Simulate larger batch
    "max_length_override": 5.0,     # Shorter audio for memory
}

TRAINING_CONFIG_A100 = {
    "batch_size": 32,               # Large batch for A100 power
    "num_epochs": 10,
    "learning_rate": 3e-5,          # Slightly higher LR for larger batches
    "num_workers": 8,               # High throughput
    "gradient_accumulation_steps": 1, # No need to accumulate
    "max_length_override": 10.0,    # Full length audio
}

# Default (fallback to conservative M2 settings)
TRAINING_CONFIG = TRAINING_CONFIG_M2_MACBOOK

# Model Names - for easy model switching
MODEL_NAMES = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
}


def get_training_config(platform: str = "auto"):
    """
    Get platform-appropriate training configuration.

    Args:
        platform: "m2", "a100", or "auto" (auto-detect)

    Returns:
        Training configuration dict
    """
    if platform == "auto":
        # Auto-detect based on available hardware
        if torch.cuda.is_available():
            # Check if it's an A100 (or similar high-end GPU)
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "a100" in gpu_name or "v100" in gpu_name or "h100" in gpu_name:
                platform = "a100"
            else:
                platform = "a100"  # Default to A100 config for any CUDA GPU
        elif torch.backends.mps.is_available():
            platform = "m2"
        else:
            platform = "m2"  # Conservative CPU fallback

    if platform == "a100":
        logger.info("Using A100/GPU training configuration")
        return TRAINING_CONFIG_A100
    else:
        logger.info("Using M2 MacBook training configuration")
        return TRAINING_CONFIG_M2_MACBOOK


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    # Only remove existing handlers if they exist
    if logger._core.handlers:
        logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
except Exception as e:
    # If there's any other loguru configuration issue, just skip it
    pass
