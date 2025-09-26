#!/bin/bash

# =============================================================================
# Triplet Deformable DETR Training Script
# =============================================================================

set -e 

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Configuration
# =============================================================================

# Training parameters
BATCH_SIZE=4
EPOCHS=40
LEARNING_RATE=0.0002
LR_DROP=20

# Paths
DATASET_PATH="/root/autodl-tmp/prostate/dataset_coco_triplet/split4"
PRETRAINED_WEIGHTS="./r50_deformable_detr-checkpoint.pth"
OUTPUT_DIR="/root/autodl-tmp/prostate/output/triplet_deformable_detr_split4_output"
TB_LOG_DIR="/root/autodl-tmp/prostate/output/tensorboard_split4_logs"

# Environment
CONDA_ENV="triplet"
PYTHONPATH_DETR="/root/detection/Deformable-DETR"

check_dataset() {
    print_info "Checking dataset path: $DATASET_PATH"
    if [ -d "$DATASET_PATH" ]; then
        if [ -f "$DATASET_PATH/train_annotations.json" ] && [ -f "$DATASET_PATH/val_annotations.json" ]; then
            print_success "Dataset found with annotation files"
        else
            print_error "Dataset directory exists but annotation files missing"
            exit 1
        fi
    else
        print_error "Dataset directory not found: $DATASET_PATH"
        exit 1
    fi
}

check_pretrained_weights() {
    print_info "Checking pretrained weights: $PRETRAINED_WEIGHTS"
    if [ -f "$PRETRAINED_WEIGHTS" ]; then
        print_success "Pretrained weights found"
    else
        print_warning "Pretrained weights not found"
        print_info "Downloading R50 Deformable DETR checkpoint..."
        wget -O "$PRETRAINED_WEIGHTS" "https://github.com/fundamentalvision/Deformable-DETR/releases/download/v1.0/r50_deformable_detr-checkpoint.pth"
        if [ $? -eq 0 ]; then
            print_success "Downloaded pretrained weights successfully"
        else
            print_error "Failed to download pretrained weights"
            exit 1
        fi
    fi
}

create_output_dirs() {
    print_info "Creating output directories"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TB_LOG_DIR"
    print_success "Output directories created"
}

run_training() {
    print_info "Starting Triplet Deformable DETR training"
    print_info "Configuration:"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Epochs: $EPOCHS"
    echo "  - Learning rate: $LEARNING_RATE"
    echo "  - LR drop: $LR_DROP"
    echo "  - Dataset: $DATASET_PATH"
    echo "  - Output: $OUTPUT_DIR"
    echo "  - TensorBoard: $TB_LOG_DIR"
    echo ""

    export PYTHONPATH="$PYTHONPATH_DETR:$PYTHONPATH"

    python triplet_main.py \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LEARNING_RATE" \
        --lr_drop "$LR_DROP" \
        --output_dir "$OUTPUT_DIR" \
        --coco_path "$DATASET_PATH" \
        --dataset_file triplet \
        --pretrained_weights "$PRETRAINED_WEIGHTS" \
        --tensorboard \
        --tb_log_dir "$TB_LOG_DIR" \
        --num_workers 4 \
        --weight_decay 0.0001 \
        --clip_max_norm 0.1

    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
        print_info "Results saved to: $OUTPUT_DIR"
        print_info "TensorBoard logs: $TB_LOG_DIR"
        print_info "To view TensorBoard: tensorboard --logdir=$TB_LOG_DIR --port=6006"
    else
        print_error "Training failed!"
        exit 1
    fi
}

main() {
    print_info "=== Triplet Deformable DETR Training Script ==="
    print_info "Starting pre-training checks..."
    
    check_dataset
    check_pretrained_weights
    create_output_dirs
    
    print_success "All checks passed! Starting training..."
    echo ""
    
    START_TIME=$(date +%s)
    
    run_training
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    print_success "=== Training Summary ==="
    printf "Training time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
    print_info "Check the output directory for trained models and logs"
    print_info "Use TensorBoard to visualize training progress:"
    print_info "  tensorboard --logdir=$TB_LOG_DIR --port=6006"
}

main "$@"
