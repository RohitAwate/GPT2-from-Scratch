import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50K BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query and value tensors, but all in one batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        """
        GPT-2 paper proposes scaling down of linear layers at the end of the block
        because they accumulate gradients.
        https://youtu.be/l8pRSuU81PU?t=4427&si=HnJnTK_piFJ1SXGm
        """
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        """
        The mask is (T, T) after tril. The attention scores tensor is (B, nh, T, T).
        The .view(1, 1, T, T) adds dummy batch and head dimensions so PyTorch can
        broadcast the mask across all batches and all heads without copying memory.

        (1, 1, T, T) broadcasts to (B, nh, T, T) automatically.

        On naming: Not really a bias, but following OpenAI/HuggingFace's naming
        convention so that we can load their weights.
        """
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # remember that self.n_embd is same as C
        hs = self.n_embd // self.n_head

        # Actual matmul is between T and hs, whereas B and nh are batch dims
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        """
        Commenting this out in favor of the Flash Attention kernel:

        # k.size(-1) just pulls its last dimension i.e. hs (scalar value)
        # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # dim=-1 because we want to calculate probabilities along the channels dimension
        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        """

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # re-shape/concat head outputs back together
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection: (B, T, C) @ (C, C) = (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Can use exact, but GPT-2 used approximate tanh GELU
        # because the exact version was slow at the time.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        """
        GPT-2 paper proposes scaling down of linear layers at the end of the block
        because they accumulate gradients.
        https://youtu.be/l8pRSuU81PU?t=4427&si=HnJnTK_piFJ1SXGm
        """
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        # This is called pre-normalization, by the way.
        # GPT-2 deviates from "Attention Is All You Need" here.
        # OG paper did post-normalization: x = layer_norm(x + attn(x))
        #
        # Pre-Normalization trains more stably, especially at greater depth + scale.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme:
        #
        # We use the same weights because in both cases we want the same behavior
        # from these matrices: similar words should have similar embeddings from
        # wte and similar words should have similar output probabilities after
        # softmax from the lm_head.
        # And of course, in the architecture they're the same shape.
        # Ref: Attention Is All You Need / Section 3.4 Embeddings and Softmax
        #
        # My latent-space interpretation:
        # We input probabilities (one hot encoding) and expect to get embeddings.
        # On the output end, we convert the same (transformed) embeddings back out
        # into a probability distribution. So it's an inverse operation, so makes
        # sense to share the weights.
        #
        # Not to mention that this also saves a ton of memory!
        self.transformer.wte.weight = self.lm_head.weight

        # This is
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        This is a callback func that gets applied to each module in an nn.Module.

        GPT-2 initializes Linear and Embedding layers with a normal distribution
        having mean=0 and std dev=0.02. Actually, they use 0.01 for wpe, but 0.02
        for wte, but Karpathy stuck with 0.02 since it doesn't make much difference.

        For biases, they zero initialize.

        Additionally, Karpathy says that standard deviation is roughly 1/sqrt(input_dim)
        of that layer. (Xavier initialization method)

        https://youtu.be/l8pRSuU81PU?t=3974&si=Rxo8XNWpS7NZTJZS
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # Twice because in each transformer block there's two elements that
                # add to the residual pathway: (1) Attention Block and (2) MLP
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of len {T}, block size is {self.config.block_size}"
        )
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, n_embd i.e. C)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (T, n_embd i.e. C)
        x = tok_emb + pos_emb

        # forward through the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # F.cross_entropy() expects input to be (C) or (N,C) if multi-dim.
            #
            # Equivalent code:
            # F.cross_entropy(
            #     input=logits.view(-1, logits.size(-1)),  # logits.size(-1) will be T
            #     target=targets.view(-1),  # this will be B*T automatically
            # )
            loss = F.cross_entropy(
                input=logits.view(B * T, -1), target=targets.view(B * T)
            )

        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        # gather all params that require grad
        param_dict = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        # separate into groups
        # any params with more than 2 dims gets weight decayed (matmul + embeddings)
        # others don't (layernorm + biases)
        decay_params = [param for param in param_dict.values() if param.dim() >= 2]
        non_decay_params = [param for param in param_dict.values() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(param.numel() for param in decay_params)
        num_non_decay_params = sum(param.numel() for param in non_decay_params)
        print(
            f"Decayed Params Tensors: {len(decay_params)} Actual Params: {num_decay_params}"
        )
        print(
            f"Non-Decayed Params Tensors: {len(non_decay_params)} Actual Params: {num_non_decay_params}"
        )
        use_fused = "cuda" in device
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataLoaderLite:
    """
    Data loading strategy: interleaved positions across processes (not shards).

    Intuitive approach: split the data into N shards, one per process, each
    loader cycling within its own shard independently.

    Karpathy's approach: all processes read from the SAME token array, but
    at interleaved offsets. Process rank r starts at position r*B*T and
    advances by B*T*num_procs each step — skipping over what the other
    processes consumed. No shard boundaries; just coordinated strides.

    Token array:
    [ P0 chunk ][ P1 chunk ][ P2 chunk ][ P3 chunk ][ P0 chunk ][ P1 chunk ] ...
     ^step 0                                          ^step 1
     pos=0                                            pos=4*B*T

    Each process sees a consistent, non-overlapping stream. P0 jumps by
    B*T*num_procs each step, landing only on its own chunks.

    This simplicity holds for single-file training. The shard-based loader
    comes later when scaling to multi-file datasets like FineWeb.
    """

    def __init__(self, B, T, proc_rank, num_procs):
        self.B = B
        self.T = T
        self.proc_rank = proc_rank
        self.num_procs = num_procs

        with open("office.txt", "r") as fd:
            text = fd.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # Each process gets its own B*T-sized shard of the data.
        # Which shard this process gets is indicated by proc_rank.
        self.current_pos = self.B * self.T * self.proc_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_pos : self.current_pos + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T * self.num_procs

        # if the next batch is gonna overflow, reset to 0
        if self.current_pos + (B * T * self.num_procs + 1) > len(self.tokens):
            self.current_pos = self.B * self.T * self.proc_rank

        return x, y


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(step: int) -> float:
    # 1. Linear warmup for warmup steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2. If beyond max_steps, use minimum LR
    if step > max_steps:
        return min_lr

    # 3. In between, use cosine decay down to min LR
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    # coeff starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    """
    simple launch: python3 train_gpt2.py
    DDP launch for e.g. 8 GPUs: torchrun --standalone --nproc_per_node=8 train_gpt2.py
    """

    import torch.distributed as dist
    from torch.distributed import destroy_process_group, init_process_group
    from torch.nn.parallel import DistributedDataParallel

    torch.manual_seed(1337)

    # See if this is a DDP run
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        # torch DDP sets environment variables about the distribution of the run
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # vanilla run, without DDP
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
            torch.mps.manual_seed(1337)
        elif torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed(1337)
        device = "cpu"  # dev override
        print(f"Using device: {device}")

    # Gradient Accumulation
    total_batch_size = 524_288  # 2^19, ~0.5M tokens
    B = 16  # micro-batch size
    T = 1024  # sequence/context length
    assert total_batch_size // (B * T * ddp_world_size) % 0 == 0, (
        "make sure total_batch_size is divisible by (B x T x ddp_world_size)"
    )
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:  # only print this once, same for all processes
        print(
            f"Total batch size: {total_batch_size} i.e. {total_batch_size / 1_000_000:.2f}M"
        )
        print(f" => Gradient Accumulation Steps: {grad_accum_steps}")

    # This uses TF32 when available (Ampere and above NVIDIA GPUs)
    torch.set_float32_matmul_precision("high")

    # Using power of 2 for vocab size for better alignment
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model: GPT = model.module if ddp else model

    optimizer = raw_model.configure_optimizer(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    loader = DataLoaderLite(B, T, proc_rank=ddp_local_rank, num_procs=ddp_world_size)

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()  # clears the accumulated gradient

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)

            # BF16 is only available on Ampere and above
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(
                    x, y
                )  # forward pass: calculates new logits and loss

            """
            We divide the loss by the number of steps. Loss functions (MSE, CrossEntropy) calculate the mean loss across batches.
            So if the network outputs BxT probabilities, the loss func has to reduce it down to one number. This is where it takes
            the mean across batches. But if we are doing micro-batches, the mean has to be calculated over grad_accum_steps, and
            not the B i.e. the micro-batch size. Thus, the normalization here.

            Karpathy: https://youtu.be/l8pRSuU81PU?t=9249&si=U3F4oexQBpg8H0Lp
            """
            loss /= grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                # Naughty way to only do gradient synchronization for just the last micro step.
                # We let the gradient accumulate for all other steps and then propagate it
                # (with DDP's averaging) in one go. Doing sync at all steps is too inefficient
                # and superfluous for our use case.
                #
                # PyTorch-sanctioned way to do this is to use "with ddp.no_sync()" but this is more concise.
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()  # propagates the loss backward, causing it to accumulate

        if ddp:
            # This is just for logging/printing the average loss across all procs.
            # loss.backwards() already calculates this internally when backpropping the loss.
            # But the local variable remains unchanged. So we want it to take on the actual
            # value that was used in backprop, so that we can accurately print it out.
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        """
        This clips the norm to a max.
        norm = sqrt(sum(param**2 for param in params))
        Remember from ML grad class? ;)
        """
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Determine and set learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()  # updates weights based on accumulated gradient
        torch.cuda.synchronize()
        t1 = time.time()

        tokens_processed = (B * T) * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed // (t1 - t0)
        if master_process:
            print(
                f"[Step {step:2d}] | Loss: {loss_accum.item():.4f} | LR: {lr:.4e} | Norm: {norm:.4f} | Time: {(t1 - t0):.2f}ms | Tokens: {tokens_per_sec} tps"
            )

    if ddp:
        destroy_process_group()

    # Skip sampling/inference
    if False:
        num_sequences = 5
        max_length = 30

        model = GPT.from_pretrained("gpt2")
        model.eval()

        """
        Using https://tiktokenizer.vercel.app/?model=gpt2
        we know that this will return 8 tokens:
        [15496, 11, 314, 1101, 257, 3303, 2746, 11]
        """
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)  # (5, 8)

        x = tokens.to(device)

        # before starting, x will be (5, 8)
        # with each generation, the 2nd i.e. index 1 dim will grow
        # capping it at max_length
        while x.size(1) < max_length:
            with torch.no_grad():
                logits = model(x)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # only grab logits of last pos
                probs = F.softmax(logits, dim=-1)  # softmax along vocab_size
                """
                Top-K sampling: Only grab the top-K probabilities, and normalize that probability distribution.
                This is what the HF model uses by default.
                top_k probs and topk_indices are both (5, 50)
                """
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # Select a token from top-K
                ix = torch.multinomial(topk_probs, 1)  # (B, 1)
                # gather the indices from topk_indices, based on ix
                xcol = torch.gather(topk_indices, -1, ix)
                # append to sequence
                x = torch.cat((x, xcol), dim=1)

        for i in range(num_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(">", decoded)


if __name__ == "__main__":
    main()
