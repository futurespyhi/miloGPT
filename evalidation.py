# Parse and visualize training logs for multiple learning rates ablation study
import os
import numpy as np
import matplotlib.pyplot as plt

sz = "124M"

learning_rates = {
    "conservative": 6e-4,
    "medium": 6e-4 * 2,
    "aggressive": 6e-4 * 3
}

loss_baseline = {"124M": 3.2924}[sz]
hella2_baseline = {"124M": 0.294463, "350M": 0.375224, "774M": 0.431986, "1558M": 0.488946}[sz]
hella3_baseline = {"124M": 0.337, "350M": 0.436, "774M": 0.510, "1558M": 0.547}[sz]

def load_log_data(log_path):
    """Load and parse log file into streams dictionary"""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        
        streams = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            step, stream, val = parts
            if stream not in streams:
                streams[stream] = {}
            streams[stream][int(step)] = float(val)
        
        # Convert to (steps[], vals[]) format for plotting
        streams_xy = {}
        for k, v in streams.items():
            xy = sorted(list(v.items()))
            if xy:  # Check if data exists
                streams_xy[k] = list(zip(*xy))
        
        return streams_xy
    except Exception as e:
        print(f"Error loading {log_path}: {e}")
        return None

# 1. Check which log files exist and load data
all_streams = {}
colors = ['blue', 'orange', 'green', 'red', 'purple']
line_styles = ['-', '--', '-.', ':', '-']

for i, (lr_name, lr_val) in enumerate(learning_rates.items()):
    log_path = f"log/log_{lr_val}.txt"
    if os.path.exists(log_path):
        print(f"Loading data for {lr_name} learning rate ({lr_val})")
        streams_data = load_log_data(log_path)
        if streams_data:
            all_streams[lr_name] = streams_data

# 2. Error handling: check if any data was loaded
if not all_streams:
    print("No valid log files found. Please run training first.")
    exit(1)

# 3. Create comparison plots
plt.figure(figsize=(16, 6))

# Panel 1: Loss comparison
plt.subplot(121)
for i, (lr_name, streams_xy) in enumerate(all_streams.items()):
    color = colors[i % len(colors)]
    line_style = line_styles[i % len(line_styles)]
    
    # Plot training loss
    if "train" in streams_xy:
        xs, ys = streams_xy["train"]
        ys = np.array(ys)
        plt.plot(xs, ys, color=color, linestyle=line_style, alpha=0.7, 
                label=f'{lr_name} train loss')
        print(f"{lr_name} - Min Train Loss: {min(ys):.4f}")
    
    # Plot validation loss
    if "val" in streams_xy:
        xs, ys = streams_xy["val"]
        ys = np.array(ys)
        plt.plot(xs, ys, color=color, linestyle=line_style, linewidth=2,
                label=f'{lr_name} val loss')
        print(f"{lr_name} - Min Val Loss: {min(ys):.4f}")

# Add baseline
if loss_baseline:
    plt.axhline(y=loss_baseline, color='r', linestyle='--', alpha=0.8,
               label=f"OpenAI GPT-2 ({sz}) baseline")

plt.xlabel("steps")
plt.ylabel("loss")
plt.yscale('log')
plt.ylim(top=4.0)
plt.legend()
plt.title("Loss Comparison")

# Panel 2: HellaSwag comparison
plt.subplot(122)
for i, (lr_name, streams_xy) in enumerate(all_streams.items()):
    color = colors[i % len(colors)]
    line_style = line_styles[i % len(line_styles)]
    
    if "hella" in streams_xy:
        xs, ys = streams_xy["hella"]
        ys = np.array(ys)
        plt.plot(xs, ys, color=color, linestyle=line_style, linewidth=2,
                label=f'{lr_name}')
        print(f"{lr_name} - Max HellaSwag: {max(ys):.4f}")

# Add baselines
if hella2_baseline:
    plt.axhline(y=hella2_baseline, color='r', linestyle='--', alpha=0.8,
               label=f"OpenAI GPT-2 ({sz})")
if hella3_baseline:
    plt.axhline(y=hella3_baseline, color='g', linestyle='--', alpha=0.8,
               label=f"OpenAI GPT-3 ({sz})")

plt.xlabel("steps")
plt.ylabel("accuracy")
plt.legend()
plt.title("HellaSwag Evaluation")

# 4. Save with appropriate filename
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
plt.tight_layout()

if len(all_streams) == 1:
    filename = f"training_results_{list(all_streams.keys())[0]}_{sz}.png"
else:
    filename = f"training_comparison_{sz}.png"

save_path = os.path.join(results_dir, filename)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figure saved to {save_path}")