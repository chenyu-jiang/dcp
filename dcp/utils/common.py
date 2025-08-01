import os

import torch
from torch.nn import ParameterDict, ParameterList


def trepr(t, verbose: bool = False, shape_only: bool = False):
    """
    Tensor and list representation
    """
    if isinstance(t, (list, tuple, ParameterList)):
        return str([trepr(e, verbose, shape_only) for e in t])
    if isinstance(t, (dict, ParameterDict)):
        return str({k: trepr(v, verbose, shape_only) for k, v in t.items()})
    if torch.is_tensor(t):
        if verbose and not shape_only:
            return str(
                (
                    t.shape,
                    t.dtype,
                    t.device,
                    t.requires_grad,
                    t.grad_fn,
                    t.is_contiguous(),
                    t.storage_offset(),
                    t._is_view(),
                    type(t),
                )
            )
        if shape_only:
            return str((t.shape, t.dtype))

        return str((t.shape, t.dtype, t.device, t.requires_grad))

    return str(t)


def read_env_int(key: str, default: int = 0) -> int:
    switch = os.environ.get(key, default)
    if isinstance(switch, str):
        return int(switch)
    return switch
