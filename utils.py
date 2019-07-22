import os

def selectGpuById(id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id)

def vars_info(var_list):
    return "\t" + "\n\t".join(["{} : {}".format(v.name, v.get_shape()) for v in var_list])