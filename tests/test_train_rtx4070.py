#!/usr/bin/env python3
"""
RTX4070 Test Version of train_gpt2.py
Reduced batch sizes and shorter training for local testing
"""

import os
import math
import time
import inspect
import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

# ------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.MILOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # use FlashAttention for optimization
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MILOGPT_SCALE_INIT = 1

    # feed forward neural network
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256     # RTX4070: reduced from 1024
    vocab_size: int = 50257   # vocabulary size: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|>
    n_layer: int = 12         # number of Transformer decoder blocks
    n_head: int = 12          # number of attention heads per block
    n_embd: int = 768         # embedding dimension (d_model in paper)

# GPT-2 model (decoder-only transformer)
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),  # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # language modeling head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'MILOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Configure AdamW optimizer with proper weight decay handling.
        """
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device_type
        print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, 
                                      fused=use_fused)
        
        return optimizer

# ------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(file):
    npt = np.load(file)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        # get the shard filenames
        data_root = "edu_fineweb10B"
        if not os.path.exists(data_root):
            print(f"WARNING: {data_root} not found. Creating dummy data for testing.")
            self.use_dummy_data = True
        else:
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            self.use_dummy_data = False
            if len(shards) == 0:
                print(f"No shards found for split {split}, using dummy data")
                self.use_dummy_data = True
        
        self.reset()

    def reset(self):
        # reset state, init at shard zero
        if self.use_dummy_data:
            # Create dummy tokens for testing
            vocab_size = 50257
            seq_len = 100000  # 100K tokens for testing
            self.tokens = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
            print("Using dummy random tokens for testing")
        else:
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T)   # targets
        
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        
        # if loading the next batch would be out of bounds, cycle back or load next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            if self.use_dummy_data:
                self.current_position = B * T * self.process_rank  # cycle back for dummy data
            else:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
        
        return x, y

# -----------------------------------------------------------------------------

# set up device (single GPU for RTX4070 testing)
ddp = False  # Disable DDP for single GPU testing
ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
master_process = True

# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Parse command line arguments
parser = argparse.ArgumentParser(description="GPT-2 RTX4070 test training")
parser.add_argument("--max_lr", type=float, default=6e-4, help="Maximum learning rate (default: 6e-4)")
args = parser.parse_args()

# set encoder
enc = tiktoken.get_encoding("gpt2")

# RTX4070 optimized batch configuration
total_batch_size = 32768   # Much smaller for RTX4070: ~32K tokens
B = 8                      # RTX4070 friendly batch size
T = 256                    # RTX4070 friendly sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print(f"RTX4070 Test Configuration:")
print(f"total desired batch size: {total_batch_size}")
print(f"micro batch size: {B}")
print(f"sequence length: {T}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304, block_size=T))
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
raw_model = model

# Learning rate schedule parameters
max_lr = args.max_lr
min_lr = max_lr * 0.1
warmup_steps = 50          # Reduced for testing
max_steps = 500            # Much shorter for testing (vs 19073 for full training)

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# Create log directory
log_dir = "log_test"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_rtx4070_test_{max_lr}.txt")
with open(log_file, "w") as f:
    pass

print("\n" + "="*50)
print("Starting RTX4070 Test Training")
print(f"Device: {device}")
print(f"Max steps: {max_steps}")
print(f"Log file: {log_file}")
print("="*50 + "\n")

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation every 50 steps (more frequent for testing)
    if step % 50 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 5  # Reduced for faster testing
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        
        print(f"step {step:3d} | validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # Training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    
    if step % 10 == 0 or last_step:  # Print every 10 steps
        print(f"step {step:3d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

print(f"\nTesting completed! Check {log_file} for results.")
print("If this runs successfully, your setup is ready for the full Lambda training.")