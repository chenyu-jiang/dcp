import torch
import torch.nn.functional as F
from dcp_flash_attn import flash_attn_varlen_kvpacked_func
from torch.utils.benchmark import Timer

from dcp.core.cost_model import AttnInstr, AttnRooflineCostModel


def run_benchmark(stmt, globals=None):
    timer = Timer(stmt=stmt, globals=globals)
    m = timer.blocked_autorange()
    return m.median


def generate_attn_range(seqlens, attn_type="causal"):
    if attn_type == "causal":
        attn_range = torch.empty(
            2, sum(seqlens), dtype=torch.int32, device="cuda"
        )
        cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(seqlens, device="cuda", dtype=torch.int32),
                0,
            ),
            (1, 0),
        ).to(torch.int32)
        for b, seqlen in enumerate(seqlens):
            attn_range[0, cu_seqlens[b] : cu_seqlens[b + 1]] = 0
            attn_range[1, cu_seqlens[b] : cu_seqlens[b + 1]] = torch.arange(
                1, seqlen + 1, device="cuda"
            )
    elif attn_type == "full":
        attn_range = torch.empty(
            2, sum(seqlens), dtype=torch.int32, device="cuda"
        )
        cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(seqlens, device="cuda", dtype=torch.int32),
                0,
            ),
            (1, 0),
        ).to(torch.int32)
        for b, seqlen in enumerate(seqlens):
            attn_range[0, cu_seqlens[b] : cu_seqlens[b + 1]] = 0
            attn_range[1, cu_seqlens[b] : cu_seqlens[b + 1]] = seqlen + 1
    else:
        raise NotImplementedError
    return attn_range


def test_model_attn_forward_full(batch_size=4, seqlen=512, attn_type="full"):
    nheads = 8
    headdim = 128
    cost_model = AttnRooflineCostModel()
    attn_range = generate_attn_range(
        [seqlen] * batch_size, attn_type=attn_type
    )
    instr = AttnInstr(
        0,
        [seqlen] * batch_size,
        [seqlen] * batch_size,
        seqlen,
        seqlen,
        attn_range.tolist(),
        None,
        None,
        None,
        None,
    )
    cost = cost_model.get_cost(instr, nheads, headdim)
    # benchmark actual function
    q = torch.randn(
        seqlen * batch_size,
        nheads,
        headdim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        seqlen * batch_size,
        2,
        nheads,
        headdim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    cu_seqlens_q = F.pad(
        torch.cumsum(
            torch.tensor(
                [seqlen] * batch_size, device="cuda", dtype=torch.int32
            ),
            0,
        ),
        (1, 0),
    ).to(torch.int32)
    cu_seqlens_kv = cu_seqlens_q.clone()

    def stmt():
        flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqlen,
            seqlen,
            attn_range=attn_range,
        )

    actual_time = run_benchmark("stmt()", globals={"stmt": stmt}) * 1e3
    print(f"Cost model: {cost} ms, Actual: {actual_time} ms")


if __name__ == "__main__":
    test_model_attn_forward_full(batch_size=4, seqlen=1024, attn_type="causal")
