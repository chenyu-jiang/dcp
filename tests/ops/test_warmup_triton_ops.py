import torch

from dcp.ops.fused_ops import warmup_triton_ops


def test_warmup_triton_ops():
    torch.cuda.set_device(0)
    warmup_triton_ops(4, 1, 128, torch.bfloat16, verbose=True)


if __name__ == "__main__":
    test_warmup_triton_ops()
