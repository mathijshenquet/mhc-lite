# One GPU, same tokens per optimizer step as the paper's 8-GPU runs (8 × 8 × 16 × 1024 = 131k).
gradient_accumulation_steps = 64
batch_size = 16
block_size = 1024
compile = True
wandb_log = False
