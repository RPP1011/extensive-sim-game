"""Text encoder for conditioning ELIT-DiT on room descriptions.

Uses sentence-transformers/all-MiniLM-L6-v2 (80MB, 384-dim) as frozen encoder.
Falls back to a simple bag-of-words embedding if sentence-transformers is unavailable.
"""

import torch
import torch.nn as nn


class FrozenSentenceEncoder(nn.Module):
    """Frozen MiniLM-L6-v2 sentence encoder. Outputs 384-dim embeddings."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        self.dim = 384
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings to (B, 384) float tensor."""
        embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        return embeddings


class SimpleTextEncoder(nn.Module):
    """Fallback: learned vocabulary embeddings with mean pooling.

    For domain-specific tactical vocabulary (~200 unique tokens).
    """

    def __init__(self, vocab_size: int = 2000, d_embed: int = 384, max_len: int = 128):
        super().__init__()
        self.dim = d_embed
        self.embed = nn.Embedding(vocab_size, d_embed)
        # Simple character-level tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size

    def tokenize(self, text: str) -> list[int]:
        """Character-level tokenization with hash-based vocab mapping."""
        tokens = []
        for ch in text.lower()[:self.max_len]:
            tokens.append(hash(ch) % (self.vocab_size - 1) + 1)  # 0 = padding
        return tokens

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to (B, d_embed) via mean pooling."""
        device = self.embed.weight.device
        B = len(texts)
        token_ids = torch.zeros(B, self.max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(B, dtype=torch.float32, device=device)

        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            L = len(tokens)
            token_ids[i, :L] = torch.tensor(tokens, dtype=torch.long)
            lengths[i] = max(L, 1)

        embedded = self.embed(token_ids)  # (B, max_len, d_embed)
        # Mean pool over non-padding tokens
        mask = (token_ids > 0).float().unsqueeze(-1)  # (B, max_len, 1)
        pooled = (embedded * mask).sum(dim=1) / lengths.unsqueeze(-1)
        return pooled


def build_text_encoder(kind: str = "minilm", device: str = "cpu") -> nn.Module:
    """Factory for text encoders."""
    if kind == "minilm":
        try:
            return FrozenSentenceEncoder(device=device)
        except ImportError:
            print("sentence-transformers not available, falling back to simple encoder")
            return SimpleTextEncoder()
    elif kind == "simple":
        return SimpleTextEncoder()
    else:
        raise ValueError(f"Unknown text encoder: {kind}")
