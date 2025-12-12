#!/bin/bash

#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Single task that will spawn multiple processes
#SBATCH --gres=gpu:4               # Request all 4 GPUs
#SBATCH --cpus-per-task=24         # Total CPUs (6 per GPU × 4)
#SBATCH --mem=256G                 # Total memory
#SBATCH --job-name=icml_benchmark
#SBATCH --output=logs/FoundationModel_ViT-B_p16_b1024_adios_benchmark_%j.out  # Single output file
#SBATCH --error=logs/FoundationModel_ViT-B_p16_b1024_adios_benchmark_%j.err   # Single error file
#SBATCH --partition=gpu            # Your specific partition
#SBATCH --signal=USR2@120          # Signal before time limit
#SBATCH --time=1-00:00:00          # Request 2 days

export NO_ALBUMENTATIONS_UPDATE=1

echo "----------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Executing script in: $(pwd)"
echo "GPU Resources: $(nvidia-smi -L)"
echo "----------------------------------------------------"

# Properly initialize and activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ssl-v1 # Make sure this matches your conda env name

# Define script arguments - Customize these as needed
CHECKPOINT_DIR="/data1/vanderbc/nandas1/FoundationModel_ViT-B_p16_b1024_adios/logs"
OUTPUT_DIR="/data1/vanderbc/nandas1/PostProc_FoundationModels/benchmark_results/FoundationModel_ViT-B_p16_b1024_adios"

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "----------------------------------------------------"

# Checkpoint processing options
INTERVAL=10000              # Process checkpoints every 10k iterations
INCLUDE_FINAL=True        # Include the final checkpoint

# Model parameters
FEATURE_DIM=768            # Feature dimension (depends on ViT type: 768 for ViT-S, 1024 for ViT-B, etc.)

# Dataset paths
MHIST_PATH="/data1/vanderbc/nandas1/Benchmarks/MHIST_patches_unnormalized"
CRC_PATH="/data1/vanderbc/nandas1/Benchmarks/CRC_unnormalized"
PCAM_PATH="/data1/vanderbc/nandas1/Benchmarks/PatchCamelyon_unnormalized"
BRACS_PATH="/data1/vanderbc/nandas1/Benchmarks/BRACS"
MIDOG_PATH="/data1/vanderbc/nandas1/Benchmarks/MiDOG++/classification/"

PANNUKE_PATH="" #/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized"
MONUSEG_PATH="" #"/data1/vanderbc/nandas1/Benchmarks/MonuSeg_patches_unnormalized"


# Dataset magnifications
PANNUKE_MAG="40x"           # 20x or 40x
MONUSEG_MAG="40x"           # 20x or 40x


# DataLoader parameters
BATCH_SIZE=64
NUM_WORKERS=4

# Augmentation parameters (only used if task-agnostic metrics are enabled)
SAMPLE_SIZE=1000
AUGS_PER_IMAGE=50
GLOBAL_SIZE=224
LOCAL_SIZE=96
N_LOCAL_CROPS=1
GLOBAL_SCALE_MIN=0.4
GLOBAL_SCALE_MAX=1.0
LOCAL_SCALE_MIN=0.05
LOCAL_SCALE_MAX=0.4

# Normalization parameters
NORM_MEAN_R=0.6816
NORM_MEAN_G=0.5640
NORM_MEAN_B=0.7232
NORM_STD_R=0.1617
NORM_STD_G=0.1714
NORM_STD_B=0.1389

# Convergence parameters (only used if task-agnostic metrics are enabled)
CONV_START_SIZE=10000
CONV_STEP_SIZE=1000
CONV_THRESHOLD=0.001
CONV_MIN_STEPS=5
CONFIDENCE_LEVEL=0.95
BOOTSTRAP_SAMPLES=20

# Classification parameters
MONTE_CARLO_ITERATIONS=10
WEIGHT_DECAY_VALUES="1e-5,1e-4,1e-3,1e-2,1e-1"
LEARNING_RATE=0.1
EARLY_STOP_PATIENCE=10
MAX_EPOCHS=100
VAL_SPLIT=0.15

# Segmentation parameters
SEG_LEARNING_RATE=5e-5
SEG_EARLY_STOP_PATIENCE=10
SEG_MAX_EPOCHS=100
SEG_VAL_SPLIT=0.2

# Random seed
SEED=42

# =============================================================================
# TASK-AGNOSTIC METRICS CONTROL
# Set to "true" to SKIP task-agnostic metrics (RankMe, CLID, Alpha-ReQ, LiDAR)
# Set to "false" or comment out to COMPUTE task-agnostic metrics
# =============================================================================
SKIP_TASK_AGNOSTIC=true

# Ensure output directory exists
mkdir -p $OUTPUT_DIR
mkdir -p logs
echo "----------------------------------------------------"

echo "Running multi-GPU benchmarking script"
if [ "$SKIP_TASK_AGNOSTIC" = "true" ]; then
    echo "Task-agnostic metrics will be SKIPPED"
else
    echo "Task-agnostic metrics will be COMPUTED"
fi
echo "----------------------------------------------------"

# Build the python command
PYTHON_CMD="python benchmarking.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --output_dir $OUTPUT_DIR \
    --interval $INTERVAL \
    --include_final $INCLUDE_FINAL \
    --feature_dim $FEATURE_DIM \
    --mhist_path \"$MHIST_PATH\" \
    --crc_path \"$CRC_PATH\" \
    --pcam_path \"$PCAM_PATH\" \
    --midog_path \"$MIDOG_PATH\" \
    --bracs_path \"$BRACS_PATH\" \
    --pannuke_path \"$PANNUKE_PATH\" \
    --monuseg_path \"$MONUSEG_PATH\" \
    --pannuke_magnification $PANNUKE_MAG \
    --monuseg_magnification $MONUSEG_MAG \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --sample_size $SAMPLE_SIZE \
    --augmentations_per_image $AUGS_PER_IMAGE \
    --global_size $GLOBAL_SIZE \
    --local_size $LOCAL_SIZE \
    --n_local_crops $N_LOCAL_CROPS \
    --global_crop_scale_min $GLOBAL_SCALE_MIN \
    --global_crop_scale_max $GLOBAL_SCALE_MAX \
    --local_crop_scale_min $LOCAL_SCALE_MIN \
    --local_crop_scale_max $LOCAL_SCALE_MAX \
    --normalize_mean_r $NORM_MEAN_R \
    --normalize_mean_g $NORM_MEAN_G \
    --normalize_mean_b $NORM_MEAN_B \
    --normalize_std_r $NORM_STD_R \
    --normalize_std_g $NORM_STD_G \
    --normalize_std_b $NORM_STD_B \
    --convergence_start_size $CONV_START_SIZE \
    --convergence_step_size $CONV_STEP_SIZE \
    --convergence_threshold $CONV_THRESHOLD \
    --convergence_min_steps $CONV_MIN_STEPS \
    --confidence_level $CONFIDENCE_LEVEL \
    --bootstrap_samples $BOOTSTRAP_SAMPLES \
    --weight_decay_values \"$WEIGHT_DECAY_VALUES\" \
    --learning_rate $LEARNING_RATE \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --max_epochs $MAX_EPOCHS \
    --val_split $VAL_SPLIT \
    --seg_learning_rate $SEG_LEARNING_RATE \
    --seg_early_stop_patience $SEG_EARLY_STOP_PATIENCE \
    --seg_max_epochs $SEG_MAX_EPOCHS \
    --seg_val_split $SEG_VAL_SPLIT \
    --monte_carlo_iterations $MONTE_CARLO_ITERATIONS \
    --seed $SEED"

# Add the skip flag if enabled
if [ "$SKIP_TASK_AGNOSTIC" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_task_agnostic"
fi

# Execute the command
eval $PYTHON_CMD

echo "----------------------------------------------------"
echo "Job has completed."
echo "----------------------------------------------------"
