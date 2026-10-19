# One GPU, same tokens per optimizer step as the paper's 8-GPU runs: 8 GPUs × 16 × 1024 = 131,072 tokens
# per step (the paper's train.py divides gradient_accumulation_steps=8 by the world size, i.e. 1 per GPU;
# on one GPU the full 8 stays). 10k steps ≈ 1.3B tokens, as the paper states.
gradient_accumulation_steps = 8
batch_size = 16
block_size = 1024
compile = True
wandb_log = False
