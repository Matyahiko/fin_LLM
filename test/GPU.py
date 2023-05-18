import torch
USE_GPU = torch.cuda.is_available()
print("USE_GPU={}".format(USE_GPU))

if USE_GPU:
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print("GPU: {}".format(torch.cuda.device_count()))
    print("GPU: {}".format(torch.cuda.get_device_properties(0).total_memory))
   