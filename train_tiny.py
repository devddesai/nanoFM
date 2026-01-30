import torch
import torch.nn.functional as F
from model import GPT, GPTConfig


#### add wandb logging
import wandb

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

# 1. Setup (Mini model for TinyShakespeare)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=128)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 2. Minimal Training Loop
model.train()
for step in range(2000):
    # X is [B, T] tokens from TinyShakespeare
    X, _ = get_batch('train') 
    B, T = X.shape
    
    # Flow Matching Logic
    x1 = model.transformer.wte(X)          # Target: Clean embeddings
    x0 = torch.randn_like(x1)              # Source: Gaussian noise
    t = torch.rand(B, device=device)       # Random time steps
    t_env = t.view(B, 1, 1)                # For broadcasting
    
    xt = (1 - t_env) * x0 + t_env * x1     # Interpolate
    v_target = x1 - x0                     # Target velocity
    
    # Forward & Backward
    v_pred = model(xt, t)
    loss = F.mse_loss(v_pred, v_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")