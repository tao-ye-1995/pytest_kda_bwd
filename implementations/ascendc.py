"""AscendC backward 的布局适配。"""

def backward(batch):
    from fla_npu.ops.ascendc import npu_chunk_kda_bwd

    packed = batch["packed"]
    def head(value):
        return (value[0].transpose(0, 1) if packed else value.transpose(1, 2)).contiguous()
    def public(value):
        return value[0] if packed else value
    result = npu_chunk_kda_bwd(
        head(batch["q"]), head(batch["k"]), head(batch["v"]), public(batch["raw_beta"]),
        head(batch["gk"]), head(batch["Aqk"]), head(batch["Akk"]), head(batch["w"]),
        head(batch["qg"]), head(batch["kg"]), head(batch["v_new"]),
        batch["h"][0] if packed else batch["h"], public(batch["grad_output"]), batch["scale"],
        raw_g=public(batch["g"]), A_log=batch["A_log"], dt_bias=batch["dt_bias"],
        cu_seqlens=batch["cu_seqlens"], chunk_indices=batch["chunk_indices"],
        chunk_size=batch["chunk_size"], safe_gate=True, lower_bound=-5.,
        use_gate_in_kernel=True, disable_recompute=True, use_exp2=True,
        q_rstd=head(batch["q_rstd"]) if batch["normalize"] else None,
        k_rstd=head(batch["k_rstd"]) if batch["normalize"] else None,
        use_beta_sigmoid_in_kernel=batch["fused_beta"],
        beta_eff=head(batch["beta"]) if batch["fused_beta"] else None,
        layout="NTD" if packed else "BNSD")
    assert len(result) == 8 and result[5] is None
    def token(value):
        return value.transpose(0, 1).unsqueeze(0).contiguous() if packed else value.transpose(1, 2).contiguous()
    return tuple(token(t) for t in result[:3]) + (
        result[3].unsqueeze(0) if packed else result[3],
        result[4].unsqueeze(0) if packed else result[4], result[6], result[7])

