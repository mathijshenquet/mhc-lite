# Deep-narrow lab model: the L model's depth (24 layers) at a tenth of the width, to reproduce depth-driven
# optimisation effects (GDSM-L gain-scalar gradients, 2026-10-20) at ~10x the speed. ~12M parameters.
out_prefix_model = "deepnarrow"
n_layer = 24
n_head = 4
n_embd = 256
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 200
weight_decay = 0.1
block_size = 1024
