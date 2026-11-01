"""Gated multimodal fusion with a two-level branch.

The proposal names three mechanisms for the detector: one shared semantic space,
attention that assigns a weight to each feature group, and a two-stage branch
that first decides real versus fake and then decides which kind of fake. This
file implements exactly those three, in the smallest form that still deserves
the names.

    BlockEncoder    one group of features -> one vector
    GatedFusion     softmax over groups -> weighted sum; the weights are the
                    attention numbers reported to the explainability stage
    TwoLevelHead    stage 1 real/fake, stage 2 human/AI among the fakes

A group that a row does not have (no image, no CLIP embedding) is masked out of
the softmax and gets its own learned "absent" vector, so "no photo" and "a
photo scoring zero" are different inputs. That distinction matters here: image
presence correlates with the label at 23% / 76% / 75%.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BlockEncoder(nn.Module):
    def __init__(self, dim_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.absent = nn.Parameter(torch.zeros(hidden))

    def forward(self, values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        encoded = self.net(values)
        return torch.where(present.unsqueeze(1), encoded, self.absent.unsqueeze(0))


class GatedFusion(nn.Module):
    """Attention over feature groups, then one head.

    `n_domains > 0` adds a bias to every gate score that depends only on the
    domain. That is the smallest change that lets a group mean different things
    in different domains: the CLIP alignment block helps on food and hurts on
    hotel, and without the bias one pooled weight has to serve both. The bias
    starts at exactly zero, so an unconditioned run is the initialisation.
    """

    def __init__(self, widths: dict[str, int], hidden: int = 64, dropout: float = 0.2,
                 n_classes: int = 3, n_domains: int = 0, domain_dim: int = 8):
        super().__init__()
        self.names = list(widths)
        self.n_domains = n_domains
        self.encoders = nn.ModuleDict(
            {name: BlockEncoder(width, hidden, dropout) for name, width in widths.items()}
        )
        self.gate = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in widths})
        if n_domains:
            self.domain_embed = nn.Embedding(n_domains, domain_dim)
            self.domain_gate = nn.ModuleDict({name: nn.Linear(domain_dim, 1)
                                              for name in widths})
            for module in self.domain_gate.values():
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.hidden = hidden

    def forward(self, blocks: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor],
                domain: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        signal = None
        if self.n_domains and domain is not None:
            signal = self.domain_embed(domain)
        hidden = []
        scores = []
        for name in self.names:
            encoded = self.encoders[name](blocks[name], masks[name])
            hidden.append(encoded)
            score = self.gate[name](encoded).squeeze(1)
            if signal is not None:
                score = score + self.domain_gate[name](signal).squeeze(1)
            # A group that is absent for this row cannot win attention.
            score = score.masked_fill(~masks[name], float("-inf"))
            scores.append(score)
        stacked = torch.stack(hidden, dim=1)              # rows x groups x hidden
        attention = torch.softmax(torch.stack(scores, dim=1), dim=1)
        # A row with no present group (cannot happen: text is always there) would
        # produce NaN; guard anyway so a broken block is loud instead of silent.
        attention = torch.nan_to_num(attention, nan=0.0)
        fused = torch.einsum("rg,rgh->rh", attention, stacked)
        return self.head(self.dropout(fused)), attention


class TwoLevelHead(nn.Module):
    """Stage 1 real/fake, stage 2 human/AI. Stage 2 sees fakes only at training."""

    def __init__(self, widths: dict[str, int], hidden: int = 64, dropout: float = 0.2,
                 n_domains: int = 0):
        super().__init__()
        self.stage1 = GatedFusion(widths, hidden, dropout, n_classes=2, n_domains=n_domains)
        self.stage2 = GatedFusion(widths, hidden, dropout, n_classes=2, n_domains=n_domains)

    def forward(self, blocks, masks, domain=None):
        logits1, attention1 = self.stage1(blocks, masks, domain)
        logits2, attention2 = self.stage2(blocks, masks, domain)
        return logits1, logits2, attention1, attention2

    @staticmethod
    def combine(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """Three-class probabilities from the two stages.

        real = P(real), human = P(fake) * P(human | fake), ai = P(fake) * P(ai | fake)
        """
        p_fake = torch.softmax(logits1, dim=1)[:, 1]
        p_kind = torch.softmax(logits2, dim=1)
        return torch.stack([1 - p_fake, p_fake * p_kind[:, 0], p_fake * p_kind[:, 1]], dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def logit(x: torch.Tensor) -> torch.Tensor:
    return torch.logit(x.clamp(1e-6, 1 - 1e-6))


__all__ = ["BlockEncoder", "GatedFusion", "TwoLevelHead", "parameter_count", "logit"]
