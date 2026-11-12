"""原始 shape_demo 和边界数据；列出用例时不导入设备库。"""
from dataclasses import dataclass
from pathlib import Path
import re

H, D, BT = 4, 128, 64


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    packed: bool = True
    normalize: bool = True
    fused_beta: bool = False
    zero_do: bool = False


def load_cases(shape_file):
    cases = []
    for line in Path(shape_file).read_text().splitlines():
        match = re.fullmatch(r"\s*cu_seq_lens_q\s*=\s*\[([\d,\s]+)\]\s*", line)
        if not match:
            raise ValueError(f"Invalid shape_demo line: {line!r}")
        cu = tuple(int(value) for value in match[1].split(","))
        if len(cu) < 2 or cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])):
            raise ValueError(f"Invalid cumulative boundaries: {cu}")
        cases.append(Case(f"shape_demo_{len(cases) + 1}", tuple(b - a for a, b in zip(cu, cu[1:]))))
    if not cases:
        raise ValueError("shape_demo is empty")
    return cases + [
        Case("edge_packed_one", (1,)),
        Case("edge_packed_tails", (1, 63, 64, 65, 129)),
        Case("edge_without_norm", (65, 129), normalize=False),
        Case("edge_sigmoid", (65, 129), fused_beta=True),
        Case("edge_dense_b2_t64", (64, 64), packed=False),
        Case("edge_dense_b1_t128", (128,), packed=False),
        Case("edge_zero_do", (65, 129), zero_do=True),
    ]


def _metadata(lengths):
    cu, indices = [0], []
    for sequence, length in enumerate(lengths):
        cu.append(cu[-1] + length)
        for chunk in range((length + BT - 1) // BT):
            indices.extend((sequence, chunk))
    return cu, indices


def make_inputs(case):
    import torch

    generator = torch.Generator().manual_seed(20261111)
    B, T = (1, sum(case.lengths)) if case.packed else (len(case.lengths), case.lengths[0])
    shape = B, T, H, D
    randn = lambda size: torch.randn(size, generator=generator)
    q, k, v = (randn(shape).bfloat16() for _ in range(3))
    if not case.normalize:
        q = torch.nn.functional.normalize(q.float(), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(k.float(), dim=-1).bfloat16()
    g = randn(shape) * .5 + torch.tensor([-4., -6., -4., -6.])[None, None, :, None]
    beta = randn(shape[:-1])
    if not case.fused_beta:
        beta = beta.sigmoid()
    inputs = dict(q=q, k=k, v=v, beta=beta, g=g,
                  A_log=randn((H,)) * .1, dt_bias=randn((H, D)) * .2)
    do = torch.zeros(shape, dtype=torch.bfloat16) if case.zero_do else randn(shape).bfloat16()
    cu, indices = _metadata(case.lengths) if case.packed else (None, None)
    options = dict(scale=D ** -.5, cu_seqlens=cu, chunk_indices=indices,
                   normalize=case.normalize, fused_beta=case.fused_beta,
                   packed=case.packed, chunk_size=BT)
    return inputs, do, options

