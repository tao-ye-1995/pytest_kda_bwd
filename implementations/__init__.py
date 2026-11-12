"""共享缓存仅在此转换一次设备和精度；各 backward 位于独立模块。"""
import torch

BF16_FIELDS = {"q", "k", "v", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h", "grad_output"}


def device_batch(cpu_cache, options, device):
    """只转换一次精度，两个设备实现共享 token-major 张量。"""
    batch = dict(options)
    batch.update({name: value.to(device=device, dtype=torch.bfloat16 if name in BF16_FIELDS else torch.float32)
                  if value is not None else None for name, value in cpu_cache.items()})
    cu, indices = options["cu_seqlens"], options["chunk_indices"]
    batch["cu_seqlens_device"] = torch.tensor(cu, device=device, dtype=torch.int64) if cu is not None else None
    batch["chunk_indices_device"] = torch.tensor(indices, device=device, dtype=torch.int64).reshape(-1, 2) \
        if indices is not None else None
    return batch

