#!/bin/bash

# Learning rate ablation study for GPT-2 pretraining
# Run 3 experiments with different learning rates

echo "====== Starting GPT-2 Learning Rate Ablation Study ======"
echo "Expected total time: ~7.5 hours on 8xA100 GPUs"
echo "=========================================================="

# Learning rates to test
LR_CONSERVATIVE=6e-4
LR_MEDIUM=1.2e-3
LR_AGGRESSIVE=1.8e-3

# Experiment 1: Conservative learning rate (baseline)
echo ""
echo "[1/3] Training with conservative learning rate: $LR_CONSERVATIVE"
echo "Expected time: ~2.5 hours"
torchrun --standalone --nproc_per_node=8 train_gpt2.py --max_lr $LR_CONSERVATIVE
if [ $? -ne 0 ]; then
    echo "ERROR: Conservative training failed!"
    exit 1
fi

# Experiment 2: Medium learning rate (2x baseline)  
echo ""
echo "[2/3] Training with medium learning rate: $LR_MEDIUM"
echo "Expected time: ~2.5 hours"
torchrun --standalone --nproc_per_node=8 train_gpt2.py --max_lr $LR_MEDIUM
if [ $? -ne 0 ]; then
    echo "ERROR: Medium training failed!"
    exit 1
fi

# Experiment 3: Aggressive learning rate (3x baseline)
echo ""
echo "[3/3] Training with aggressive learning rate: $LR_AGGRESSIVE"  
echo "Expected time: ~2.5 hours"
torchrun --standalone --nproc_per_node=8 train_gpt2.py --max_lr $LR_AGGRESSIVE
if [ $? -ne 0 ]; then
    echo "ERROR: Aggressive training failed!"
    exit 1
fi

echo ""
echo "====== All experiments completed successfully! ======"
echo "Generated log files:"
ls -la log/log_*.txt
echo "Download log/ folder to run evaluation locally"
echo "================================================="