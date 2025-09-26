#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Default paths
DATASET_PATH="/root/autodl-tmp/prostate/dataset_coco_triplet/split3"
MODEL_PATH="/root/autodl-tmp/prostate/output/triplet_deformable_detr_split3_output/checkpoint0008.pth"
OUTPUT_DIR="./inference_results_split3"
PREDICTIONS_FILE="test_predictions.json"
SAVE_YOLO=true
YOLO_DIR="labels"
CONF_THRESHOLD="0.5"

# Environment
PYTHONPATH_DETR="/ssd/prostate/detection/Deformable-DETR"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model PATH        Path to trained model checkpoint"
    echo "                      (default: ./triplet_deformable_detr_output/checkpoint.pth)"
    echo "  --dataset PATH      Path to dataset directory"
    echo "                      (default: /ssd/prostate/prostate_track_v2/dataset_coco_triplet/split1)"
    echo "  --output DIR        Output directory for results"
    echo "                      (default: ./inference_results)"
    echo "  --predictions FILE  Predictions filename"
    echo "                      (default: test_predictions.json)"
    echo "  --yolo              Also save predictions in YOLO format"
    echo "  --yolo-dir NAME     YOLO labels directory name (default: labels)"
    echo "  --conf THRESHOLD    Confidence threshold for saving predictions (default: 0.5)"
    echo "                      Note: mAP calculation always uses all predictions"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default paths"
    echo "  $0 --model ./best_model.pth          # Use specific model"
    echo "  $0 --output ./my_results             # Custom output directory"
    echo "  $0 --yolo                            # Also generate YOLO format"
    echo "  $0 --yolo --yolo-dir my_labels       # Custom YOLO directory"
    echo "  $0 --conf 0.3                       # Lower confidence threshold"
    echo "  $0 --yolo --conf 0.7                # YOLO format with higher threshold"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --predictions)
            PREDICTIONS_FILE="$2"
            shift 2
            ;;
        --yolo)
            SAVE_YOLO=true
            shift
            ;;
        --yolo-dir)
            YOLO_DIR="$2"
            shift 2
            ;;
        --conf)
            CONF_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

check_model() {
    print_info "Checking model checkpoint: $MODEL_PATH"
    if [ -f "$MODEL_PATH" ]; then
        print_success "Model checkpoint found"
    else
        print_error "Model checkpoint not found: $MODEL_PATH"
        print_info "Available checkpoints in ./triplet_deformable_detr_output/:"
        ls -la ./triplet_deformable_detr_output/*.pth 2>/dev/null || echo "  No .pth files found"
        exit 1
    fi
}

check_dataset() {
    print_info "Checking test dataset: $DATASET_PATH"
    if [ -d "$DATASET_PATH" ]; then
        if [ -f "$DATASET_PATH/test_annotations.json" ] && [ -d "$DATASET_PATH/test" ]; then
            print_success "Test dataset found"
        else
            print_error "Test dataset incomplete. Missing test/ directory or test_annotations.json"
            print_info "Expected structure:"
            echo "  $DATASET_PATH/"
            echo "  ├── test/"
            echo "  └── test_annotations.json"
            exit 1
        fi
    else
        print_error "Dataset directory not found: $DATASET_PATH"
        exit 1
    fi
}

run_inference() {
    print_info "=== Starting Triplet Deformable DETR Inference ==="
    print_info "Configuration:"
    echo "  - Model: $MODEL_PATH"
    echo "  - Dataset: $DATASET_PATH"
    echo "  - Output: $OUTPUT_DIR"
    echo "  - Predictions: $PREDICTIONS_FILE"
    if [ "$SAVE_YOLO" = true ]; then
        echo "  - YOLO labels: $YOLO_DIR/"
    fi
    echo ""

    mkdir -p "$OUTPUT_DIR"

    START_TIME=$(date +%s)

    CMD_ARGS=(
        --model_path "$MODEL_PATH"
        --coco_path "$DATASET_PATH"
        --dataset_file triplet
        --output_dir "$OUTPUT_DIR"
        --predictions_file "$PREDICTIONS_FILE"
        --conf_threshold "$CONF_THRESHOLD"
        --batch_size 1
        --num_workers 4
    )

    if [ "$SAVE_YOLO" = true ]; then
        CMD_ARGS+=(--save_yolo_labels)
        CMD_ARGS+=(--yolo_labels_dir "$YOLO_DIR")
    fi

    python triplet_inference.py "${CMD_ARGS[@]}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    if [ $? -eq 0 ]; then
        print_success "Inference completed successfully!"
        printf "Inference time: %02d:%02d\n" $MINUTES $SECONDS
        print_info "Results saved to: $OUTPUT_DIR"
        print_info "Predictions file: $OUTPUT_DIR/$PREDICTIONS_FILE"
        
        if [ -f "$OUTPUT_DIR/$PREDICTIONS_FILE" ]; then
            PRED_SIZE=$(wc -l < "$OUTPUT_DIR/$PREDICTIONS_FILE")
            print_info "Predictions file size: $PRED_SIZE lines"
        fi
    else
        print_error "Inference failed!"
        exit 1
    fi
}


main() {
    print_info "=== Triplet Deformable DETR Inference ==="
    
    check_dataset
    check_model
    
    print_success "All checks passed! Starting inference..."
    echo ""
    
    run_inference
    
    print_success "=== Inference Complete ==="
}

main "$@"
