import time
import os
import GPUtil


def select_devices(num_gpus_to_use):

    if num_gpus_to_use == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    else:
        gpu_to_use = GPUtil.getAvailable(
            order="first",
            limit=num_gpus_to_use,
            maxLoad=0.15,
            maxMemory=0.15,
            includeNan=False,
            excludeID=[],
            excludeUUID=[],
        )
        if len(gpu_to_use) < num_gpus_to_use:
            print("Couldnt find enough GPU(s), waiting and retrying in 15 seconds")
            time.sleep(15)
            return select_devices(num_gpus_to_use=num_gpus_to_use)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_idx) for gpu_idx in gpu_to_use])

        print("GPUs selected have IDs {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
