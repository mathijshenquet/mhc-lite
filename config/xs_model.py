# XS: a complete small-scale experiment (not a truncated S run). ~5M non-embedding params,
# ~100M tokens ≈ 20 tokens/param, full cosine schedule. One GPU.
wandb_group = "xs"
out_prefix_model = "xs"

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
block_size = 512

batch_size = 32
gradient_accumulation_steps = 2     # 2 × 32 × 512 = 32k tokens / step
max_iters = 3000
lr_decay_iters = 3000
warmup_iters = 100
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
eval_interval = 500
eval_iters = 100
compile = True
wandb_log = False
