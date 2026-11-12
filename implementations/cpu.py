"""Independent KDA recurrence. Only PyTorch tensor mathematics; no device kernels.

Inputs are token-major [B, T, H, D]. With cu_seqlens, B must be one and
each interval starts with a zero state. Gate values are natural log decays:
    log_decay = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    S_decay = exp(log_decay)[..., None] * S
    residual = v - k^T S_decay
    S = S_decay + beta * k * residual^T
    o = scale * q^T S
Optional q/k normalization and beta sigmoid belong to this mathematical
function, so autograd differentiates the original inputs, including both.
"""
from __future__ import annotations

import torch

INPUT_NAMES = ("q", "k", "v", "beta", "g", "A_log", "dt_bias")
GRAD_NAMES = ("dq", "dk", "dv", "dbeta", "dg", "dA", "dbias")


def kda_recurrence(q, k, v, beta, g, A_log, dt_bias, *, scale,
                   cu_seqlens=None, normalize=True, fused_beta=False,
                   lower_bound=-5.0, eps=1e-6):
    """Evaluate the unchunked recurrence without rounding intermediates."""
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + eps)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + eps)
    if fused_beta:
        beta = beta.sigmoid()
    log_decay = lower_bound * torch.sigmoid(A_log.exp()[None, None, :, None] * (g + dt_bias))
    B, T, H, K = q.shape
    V = v.shape[-1]
    intervals = ([(0, int(a), int(b)) for a, b in zip(cu_seqlens, cu_seqlens[1:])]
                 if cu_seqlens is not None else [(b, 0, T) for b in range(B)])
    if cu_seqlens is not None:
        assert B == 1 and cu_seqlens[0] == 0 and cu_seqlens[-1] == T
    outputs = []
    for batch, start, end in intervals:
        state = q.new_zeros(H, K, V)
        for token in range(start, end):
            decayed = state * log_decay[batch, token].exp().unsqueeze(-1)
            kt = k[batch, token]
            residual = v[batch, token] - (kt.unsqueeze(-1) * decayed).sum(-2)
            state = decayed + (beta[batch, token, :, None] * kt).unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append(scale * (q[batch, token].unsqueeze(-1) * state).sum(-2))
    # Keep all seven inputs in the graph even when every sequence is empty.
    zero = sum(x.sum() * 0 for x in (q, k, v, beta, g, A_log, dt_bias))
    return ((torch.stack(outputs).reshape(B, T, H, V) if outputs
             else v.new_empty(B, T, H, V)) + zero)


def mathematical_value_and_grad(inputs, do, **options):
    """Gold: same quantized input values promoted to CPU FP64, FP64 autograd."""
    leaves = [inputs[name].detach().to(device="cpu", dtype=torch.float64).requires_grad_(True)
              for name in INPUT_NAMES]
    output = kda_recurrence(*leaves, **options)
    loss = (output * do.detach().to(device="cpu", dtype=torch.float64)).sum()
    gradients = torch.autograd.grad(loss, leaves)
    return output.detach(), tuple(x.detach() for x in gradients)


def cpu_forward_cache_and_backward(inputs, do, *, scale, cu_seqlens=None,
                                   normalize=True, fused_beta=False, chunk_size=64):
    """纯 CPU FP64 数学前向缓存和 autograd 反向；不调用任何设备实现。"""
    leaves = [inputs[name].detach().to("cpu", torch.float64).requires_grad_(True)
              for name in INPUT_NAMES]
    raw_q, raw_k, v, raw_beta, g, a_log, bias = leaves
    B, T, H, K = raw_q.shape
    V, BT = v.shape[-1], chunk_size
    qr = torch.rsqrt(raw_q.square().sum(-1) + 1e-6) if normalize else None
    kr = torch.rsqrt(raw_k.square().sum(-1) + 1e-6) if normalize else None
    q = raw_q * qr[..., None] if normalize else raw_q
    k = raw_k * kr[..., None] if normalize else raw_k
    beta = raw_beta.sigmoid() if fused_beta else raw_beta
    gate = -5.0 * torch.sigmoid(a_log.exp()[None, None, :, None] * (g + bias))
    intervals = ([(0, int(a), int(b)) for a, b in zip(cu_seqlens, cu_seqlens[1:])]
                 if cu_seqlens is not None else [(b, 0, T) for b in range(B)])
    chunks, groups = [], []
    for batch, start, end in intervals:
        first = len(chunks)
        chunks.extend((batch * T + pos, min(BT, end - pos)) for pos in range(start, end, BT))
        groups.append((first, len(chunks)))
    starts = torch.tensor([pos for pos, _ in chunks], dtype=torch.int64)
    lengths = torch.tensor([length for _, length in chunks], dtype=torch.int64)
    index = starts[:, None] + torch.arange(BT)[None, :]
    valid = torch.arange(BT)[None, :] < lengths[:, None]
    safe_index = index.clamp(max=B * T - 1)

    def chunked(value):
        tail = value.shape[2:]
        gathered = value.reshape(B * T, *tail)[safe_index]
        mask = valid.reshape(*valid.shape, *((1,) * len(tail)))
        return (gathered * mask).transpose(1, 2)

    def tokens(value):
        rows = value.transpose(1, 2)[valid]
        flat = value.new_zeros((B * T, *rows.shape[1:]))
        return flat.index_copy(0, index[valid], rows).reshape(B, T, *rows.shape[1:])

    qc, kc, vc, bc = (chunked(x) for x in (q, k, v, beta))
    cumulative = chunked(gate).cumsum(-2)
    positive, negative = cumulative.exp(), (-cumulative).exp()
    qg, k_positive = qc * positive, kc * positive
    k_negative = kc * negative
    interaction = (k_positive @ k_negative.transpose(-1, -2)) * bc[..., None]
    system = torch.tril(interaction, diagonal=-1) + torch.eye(BT, dtype=torch.float64)
    inverse = torch.linalg.solve_triangular(
        system, torch.eye(BT, dtype=torch.float64).expand_as(system), upper=False)
    # L^-1 diag(beta) 是残差更新系数；Akk 保存 L^-1，beta 单独作为输入。
    w = inverse @ (k_positive * bc[..., None])
    u = inverse @ (vc * bc[..., None])
    aqk = torch.tril((qg @ k_negative.transpose(-1, -2)) * scale)
    last = cumulative[torch.arange(len(chunks)), :, lengths - 1]
    kg = kc * (last[:, :, None, :] - cumulative).exp()
    # unbind 的反向统一收集各块梯度，避免逐次 select 反向分配完整大张量。
    qw, ww, uw, aw, kw, lw = (value.unbind(0) for value in (qg, w, u, aqk, kg, last))
    states, updates, outputs = [], [], []
    for first, end in groups:
        state = q.new_zeros(H, K, V)
        for ic in range(first, end):
            states.append(state)
            vn = uw[ic] - ww[ic] @ state
            updates.append(vn)
            outputs.append(scale * (qw[ic] @ state) + aw[ic] @ vn)
            state = lw[ic].exp()[..., None] * state + kw[ic].transpose(-1, -2) @ vn
    output = tokens(torch.stack(outputs))
    gradients = list(torch.autograd.grad((output * do.double()).sum(), leaves))
    # 零初始状态使每段第一个 token 的 gate 不影响任何输出，这是精确数学性质。
    gradients[4] = gradients[4].clone()
    for batch, start, end in intervals:
        if start < end:
            gradients[4][batch, start] = 0
    if all(end - start <= 1 for _, start, end in intervals):
        gradients[5], gradients[6] = torch.zeros_like(a_log), torch.zeros_like(bias)
    h = torch.stack(states)
    if cu_seqlens is not None:
        h = h.unsqueeze(0)
    else:
        h = h.reshape(B, T // BT, H, K, V)
    cache = dict(q=q, k=k, v=v, beta=beta, raw_beta=raw_beta, g=g,
                 A_log=a_log, dt_bias=bias, q_rstd=qr, k_rstd=kr,
                 gk=tokens(cumulative / torch.log(torch.tensor(2., dtype=torch.float64))),
                 Aqk=tokens(aqk), Akk=tokens(inverse), w=tokens(w), u=tokens(u),
                 qg=tokens(qg), kg=tokens(kg), v_new=tokens(torch.stack(updates)), h=h,
                 grad_output=do.double())
    return {name: value.detach() if value is not None else None for name, value in cache.items()}, \
        output.detach(), tuple(value.detach() for value in gradients)
