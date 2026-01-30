"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("FLASH ATTENTION NOT AVAILABLE! you gotta write the attention yourself :(")

    def forward(self, x):
        B, T, C = x.size() 
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, dropout_p=self.dropout if self.training else 0)

        # reassemble all heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class AdaptiveLayerNorm(nn.Module):
    """ LayerNorm with time step embeddings """

    def __init__(self, n_embd, eps=1e-6):
        super.__init__()
        # elementwise affine = False bc we provide scale/shift from t
        self.ln = nn.LayerNorm(n_embd, elementwise_affine=False, eps=eps)
    
    def forward(self, x, scale, shift):
        x = self.ln(x)

        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = AdaptiveLayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = AdaptiveLayerNorm(config.n_embd)
        self.mlp = MLP(config)

        self.adaLN_Modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.n_embd, 6*config.n_embd))

    def forward(self, x, t_emb):
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_Modulation(t_emb).chunk(6, dim=1)

        # attention path
        norm_x1 = self.ln_1(x, scale_msa, shift_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm_x1)

        # mlp path
        norm_x2 = self.ln_2(x, scale_mlp, shift_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x2)

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = AdaptiveLayerNorm(config.n_embd),
        ))

        self.time_embedder = TimeEmbedder(config.n_embd)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.n_embd, 2*config.n_embd))

        # Velocity head: maps back to embedding space for Flow Matching
        self.v_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # dont need weight tying from token to embedding from embedding to tokens bc we are predicting velocity

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        for block in self.transformer.h:
            nn.init.zeros_(block.adaLN_Modulation[-1].weight)
            nn.init.zeros_(block.adaLN_Modulation[-1].bias)

        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, t, targets=None):
        device = x.device
        b, t_seq, _ = x.size()
        assert t_seq <= self.config.block_size, f"Cannot forward sequence of length {t_seq}, block size is only {self.config.block_size}"
        
        t_emb = self.time_embedder(t) # shape: [B, n_embd]
        pos = torch.arange(0, t_seq, dtype=torch.long, device=device)

        # forward the GPT model itself
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(x + pos_emb)

        for block in self.transformer.h:
            x = block(x, t_emb)
        
        scale, shift = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.transformer.ln_f(x, scale, shift)

        v_pred = self.v_head(x)
        loss = None

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.mse_loss(v_pred, targets)
        
        return v_pred, loss
    
    @torch.no_grad()
    def sample(self, x_init, steps=50):
        '''
        Simple Euler sampler for inference
        x_init: initial noisy input (B, T, C)
        '''
        self.eval()
        xt = x_init
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = i * dt
            t = torch.full((x_init.shape[0],), t_curr, device=x_init.device)
            v, _ = self.forward(xt, t)
            xt = xt + v * dt
        return xt
    
    @torch.no_grad()
    def generate_tokens(self, x_init, steps=50):
        # 1. Run the flow to get clean embeddings
        x_gen = self.sample(x_init, steps) # Shape [B, T, C]
        
        # 2. Project back to vocabulary (Dot product / Cosine similarity)
        # x_gen: [B, T, C], wte.weight: [Vocab, C]
        # We calculate the similarity score for every token in the vocab
        logits = x_gen @ self.transformer.wte.weight.T # Shape [B, T, Vocab]
        
        # 3. Take the most likely token (the nearest neighbor)
        tokens = torch.argmax(logits, dim=-1) # Shape [B, T]
        return tokens
    



    ######################################################
    #### some extra stuff that i won't need as of now ####
    #######################################################

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
