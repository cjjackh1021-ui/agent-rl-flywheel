"""Pure-PyTorch lightweight Transformer backend for fast Mac verification and unit tests."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_rl_flywheel.env.proxy import TurnSample


class SimpleByteTokenizer:
    """Lossless UTF-8 byte-level tokenizer supporting arbitrary Unicode/JSON without OOVs."""

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    vocab_size: int = 259  # 3 special tokens + 256 byte values

    def encode(self, text: str) -> list[int]:
        """Encode any UTF-8 string into integer token IDs `[3..258]`."""
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        """Decode integer token IDs back to UTF-8 text, ignoring special tokens."""
        raw = bytes(tid - 3 for tid in token_ids if 3 <= tid < self.vocab_size)
        return raw.decode("utf-8", errors="replace")


class _CausalSelfAttention(nn.Module):
    """Minimal multi-head causal self-attention using PyTorch SDPA (zero warnings)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.out(attn)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCausalLM(nn.Module):
    """Lightweight 2-layer causal Transformer for local Mac PyTorch training & testing."""

    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        if seqlen > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            seqlen = input_ids.shape[1]
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.ln_f(h))


class TinyCausalLMBackend:
    """Backend wrapping `TinyCausalLM` + `SimpleByteTokenizer` + frozen reference snapshot."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 2048,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.tokenizer = SimpleByteTokenizer()
        self.max_seq_len = max_seq_len
        self.model = TinyCausalLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        ).to(self.device)
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def sync_reference(self) -> None:
        """Snapshot current actor weights into the frozen reference model."""
        self.ref_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.ref_model.eval()

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward_logits(self, input_ids: torch.Tensor, use_ref: bool = False) -> torch.Tensor:
        target = self.ref_model if use_ref else self.model
        return target(input_ids.to(self.device))

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    def generate_with_logprobs(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> TurnSample:
        """Autoregressively sample tokens and record exact per-token rollout logprobs."""
        ctx_ids = list(prompt_token_ids[-(self.max_seq_len - max(1, max_new_tokens)) :])
        if not ctx_ids:
            ctx_ids = [self.tokenizer.bos_token_id]
        generated: list[int] = []
        gen_logprobs: list[float] = []

        with self.inference_mode(), torch.no_grad():
            cur = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
            for _ in range(max_new_tokens):
                logits = self.model(cur)[:, -1, :]
                log_probs = F.log_softmax(logits / max(1e-4, temperature), dim=-1) if temperature > 0 else F.log_softmax(logits, dim=-1)
                if temperature <= 0.0:
                    next_id = int(torch.argmax(log_probs, dim=-1).item())
                else:
                    probs = torch.exp(log_probs)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                lp = float(log_probs[0, next_id].item())
                if next_id == self.tokenizer.eos_token_id:
                    break
                generated.append(next_id)
                gen_logprobs.append(lp)
                next_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                cur = torch.cat([cur, next_t], dim=1)
                if cur.shape[1] >= self.max_seq_len:
                    break

        text = self.decode(generated)
        return TurnSample(text=text, token_ids=generated, logprobs=gen_logprobs)

    def save(self, out_dir: str) -> None:
        from pathlib import Path

        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "tiny_causal_lm.pt")
