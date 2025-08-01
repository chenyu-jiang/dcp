# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import gc


def reserve_full_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory
    # try to allocate all memory
    while True:
        try:
            t = torch.empty(
                total_memory,
                dtype=torch.uint8,
                device=torch.cuda.current_device(),
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                total_memory -= 128 * 1024 * 1024  # 128MB
                continue
            else:
                # not an OOM error
                raise e
        break
    current_memory = torch.cuda.memory_allocated()
    del t
    gc.collect()
    # # free the memory
    # torch.cuda.empty_cache()
    # # allocate again, but reduce 1GB for other stuff
    # total_memory = int(total_memory - 1e9)
    # t = torch.empty(
    #     total_memory, dtype=torch.uint8, device=torch.cuda.current_device()
    # )
    # current memory should be higher than the maximum
    # memory limit of the device. currently adjusted by hand
    # but may be automated.
    return total_memory, current_memory
