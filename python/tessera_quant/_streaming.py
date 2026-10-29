# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Layer-local checkpoint transforms, preserving RFC-0005's float64 arithmetic."""

from collections.abc import Iterator, Mapping
from fractions import Fraction

import torch

from tessera_quant.checkpoint import FloatModel
from tessera_quant.migration import transform

_CHANGED = frozenset({"q", "k", "v", "attn_out", "up", "h"})


def transformed_checkpoint(
    model: FloatModel, maxima: Mapping[str, torch.Tensor], alpha: Fraction
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Lazily transforms a checkpoint, caching one group and its changed statistics.

    Each transformer block is one group, with separate embedding and final-norm/LM-head
    groups. RFC-0005's folding, equalization and migration have no dependencies between these
    groups. Unchanged channel statistics remain resident; transformed weights are discarded
    before reading another group. A caller can still retain a tensor it explicitly borrows.
    """
    groups = _Groups(model, maxima, alpha)
    return _Weights(groups), _Maxima(groups)


class _Groups:
    def __init__(
        self, model: FloatModel, maxima: Mapping[str, torch.Tensor], alpha: Fraction
    ) -> None:
        self.model = model
        self.maxima = maxima
        self.alpha = alpha
        self.names: dict[str, list[str]] = {}
        self.group: dict[str, str] = {}
        for name in model.names:
            group = ".".join(name.split(".")[:3]) if name.startswith("model.layers.") else name
            if name in {"model.norm.weight", "lm_head.weight"}:
                group = "final"
            self.names.setdefault(group, []).append(name)
            self.group[name] = group
        self.stat_group: dict[str, str] = {}
        self.stat_names: dict[str, list[str]] = {}
        for name in maxima:
            fields = name.split(".")
            if len(fields) == 3 and fields[0] == "layers" and fields[2] in _CHANGED:
                group = f"model.layers.{fields[1]}"
                self.stat_group[name] = group
            if len(fields) == 3 and fields[0] == "layers":
                self.stat_names.setdefault(f"model.layers.{fields[1]}", []).append(name)
        self.current: str | None = None
        self.weights: dict[str, torch.Tensor] = {}
        self.statistics: dict[str, torch.Tensor] = {}

    def load(self, group: str) -> None:
        if self.current == group:
            return
        # Release the previous group before even reading the next one's BF16 source tensors.
        self.current = None
        self.weights = {}
        self.statistics = {}
        source = {name: self.model.tensor(name) for name in self.names[group]}
        maxima = {name: self.maxima[name] for name in self.stat_names.get(group, ())}
        self.weights, self.statistics = transform(self.model.arch, source, maxima, self.alpha)
        self.current = group


class _Weights(Mapping[str, torch.Tensor]):
    def __init__(self, groups: _Groups) -> None:
        self.groups = groups

    def __getitem__(self, name: str) -> torch.Tensor:
        self.groups.load(self.groups.group[name])
        return self.groups.weights[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.groups.model.names)

    def __len__(self) -> int:
        return len(self.groups.model.names)


class _Maxima(Mapping[str, torch.Tensor]):
    def __init__(self, groups: _Groups) -> None:
        self.groups = groups

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self.groups.stat_group:
            return self.groups.maxima[name].to(torch.float64)
        self.groups.load(self.groups.stat_group[name])
        return self.groups.statistics[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.groups.maxima)

    def __len__(self) -> int:
        return len(self.groups.maxima)
