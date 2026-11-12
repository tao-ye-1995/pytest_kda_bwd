"""直接 import 本地 FLA backward，并补齐归一化和 sigmoid 的链式梯度。"""
import importlib
from pathlib import Path
import sys

def backward(batch):
    root = Path(batch["fla_root"]).resolve()
    assert (root / "fla/ops/kda/chunk_bwd.py").is_file(), f"Missing local FLA source: {root}"
    sys.path.insert(0, str(root))
    module = importlib.import_module("fla.ops.kda.chunk_bwd")
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
    from fla.backends import _load_operation_registry
    from fla.modules.norm.l2norm import l2norm_bwd
    from fla.ops.common.gate import fused_beta_sigmoid_bwd

    backend = _load_operation_registry("kda").get_active()
    assert backend is not None and backend.backend_type == "triton_ascend"
    result = module.chunk_kda_bwd(
        q=batch["q"], k=batch["k"], v=batch["v"], beta=batch["beta"],
        Aqk=batch["Aqk"], Akk=batch["Akk"], scale=batch["scale"], initial_state=None,
        do=batch["grad_output"], dht=None, g=batch["gk"], g_org=batch["g"],
        state_v_first=False, cu_seqlens=batch["cu_seqlens_device"],
        chunk_indices=batch["chunk_indices_device"], chunk_size=batch["chunk_size"],
        safe_gate=True, lower_bound=-5., use_gate_in_kernel=True,
        A_log=batch["A_log"], dt_bias=batch["dt_bias"].reshape(-1), disable_recompute=True,
        w=batch["w"], u=batch["u"], qg=batch["qg"], kg=batch["kg"],
        v_new=batch["v_new"], h=batch["h"])
    dq, dk, dv, dbeta, dg, dh0, dA, dbias = result
    assert dh0 is None
    if batch["normalize"]:
        dq = l2norm_bwd(batch["q"], batch["q_rstd"], dq)
        dk = l2norm_bwd(batch["k"], batch["k_rstd"], dk)
    if batch["fused_beta"]:
        dbeta = fused_beta_sigmoid_bwd(x=batch["raw_beta"], dy=dbeta, scale=1.)
    return dq.bfloat16(), dk.bfloat16(), dv.bfloat16(), dbeta.float(), dg.float(), dA.float(), dbias.reshape_as(batch["dt_bias"]).float()

