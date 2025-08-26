#!/bin/bash

# GPT-2 learning rate ablation study
# Tests 3 different learning rates with fault tolerance

echo "====== GPT-2 Learning Rate Ablation Study ======"
echo "Expected total time: ~7.5 hours on 8xA100 GPUs"
echo "================================================"

# Define learning rates
LR_RATES=(6e-4 1.2e-3 1.8e-3)
LR_NAMES=("Conservative" "Medium" "Aggressive")

# Run experiments
for i in {0..2}; do
    echo ""
    echo "[$((i+1))/3] Training with ${LR_NAMES[i]} learning rate: ${LR_RATES[i]}"
    echo "Expected time: ~2.5 hours"
    
    # Run training with fault tolerance
    torchrun --standalone --nproc_per_node=8 train_gpt2.py --max_lr ${LR_RATES[i]} || \
        echo "Warning: Training completed with non-zero exit code, continuing..."
done

# Display results
echo ""
echo "====== All experiments completed! ======"
echo "Generated log files:"
ls -la log/log_*.txt 2>/dev/null || echo "No log files found"
echo "Download log/ folder to run evaluation locally"
echo "=============================================="