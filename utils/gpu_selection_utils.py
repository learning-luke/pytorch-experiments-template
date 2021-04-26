import time
import os
import GPUtil
from rich import print

# explicit by pass of the below


def select_devices(num_gpus_to_use, max_load, max_memory, exclude_gpu_ids):

    if num_gpus_to_use == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
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
                "Couldnt find enough GPU(s) as required by the user, stopping program "
                "- consider reducing "
                "the requirements or using num_gpus_to_use=0 to use CPU"
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_idx) for gpu_idx in gpu_to_use
        )

        print("GPUs selected have IDs {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
