import time
import os
import GPUtil
import torch

# explicit by pass of the below


def select_devices(
    num_gpus_to_use, max_load=0.01, max_memory=0.01, exclude_gpu_ids=[], gpu_to_use=None
):

    if num_gpus_to_use == 0 or not torch.cuda.is_available():
        return ""
    elif gpu_to_use is None:
        gpu_to_use = GPUtil.getAvailable(
            order="first",
            limit=num_gpus_to_use,
            maxLoad=max_load,
            maxMemory=max_memory,
            includeNan=False,
            excludeID=exclude_gpu_ids,
            excludeUUID=[],
        )
        if len(gpu_to_use) < num_gpus_to_use:
            raise OSError(
                "Couldnt find enough GPU(s) as required by the user, stopping program - consider reducing "
                "the requirements or using num_gpus_to_use=0 to use CPU"
            )

    return [str(gpu_idx) for gpu_idx in gpu_to_use]
