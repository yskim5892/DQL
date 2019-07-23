import os
import json

def selectGpuById(id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id)

def vars_info(var_list):
    return "\t" + "\n\t".join(["{} : {}".format(v.name, v.get_shape()) for v in var_list])

def toJson(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=True, indent=4)

def create_dir(dirname):
    if not os.path.exists(dirname):
        print("Creating %s"%dirname)
        os.makedirs(dirname)
    else:
        print("Already %s exists"%dirname)

def create_muldir(*args):
    for dirname in args:
        create_dir(dirname)


def params2id(*args):
    nargs = len(args)
    id_ = '{}'+'_{}'*(nargs-1)
    return id_.format(*args)

class NameFormat:
    def __init__(self, *attrs):
        self.attrs = attrs
        self.nattr = len(self.attrs)
        for attr in self.attrs:
            assert type(attr)==str, "Type of attributes should be sting"
    def get_id_from_args(self, args):
        return params2id(*tuple([getattr(args, attr) for attr in self.attrs]))

    def get_query_file_id_from_args(self, args, invariables):
        return params2id(*tuple(['*' if not attr in invariables else getattr(args, attr) for attr in self.attrs]))

    def update_args_with_id(self, args, id_):
        id_split = id_split('_')
        assert len(id_split)==self.nattr, "The number of components of id_ and the number of attributes should be same"

        for i in range(self.nattr):
            attr = self.attrs[idx]
            type_attr = type(getattr(args, attr))
            setattr(args, attr, type_attr(id_split[idx]))


