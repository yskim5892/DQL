import os
import json
import math
import tensorflow as tf

def queue_smart_put(q, item, maxsize):
    if len(q) >= maxsize:
        q.pop(0)
    q.append(item)

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

class SummaryWriter:
    def __init__(self, save_path):
        self.writer = tf.summary.FileWriter(save_path)

    def add_summary(self, tag, simple_value, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)])
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, dict_, global_step):
        for key in dict_.keys():
            self.add_summary(str(key), dict_[key], global_step)

def dist_point_line_passing_two_points(x0, x1, xd):
    if (x0[0] == x1[0] and x0[1] == x1[1]):
        return math.sqrt((x0[0] - xd[0])**2 + (x0[1] - xd[1])**2)

    d2_0d = (x0[0] - xd[0])**2 + (x0[1] - xd[1])**2
    d2_1d = (x1[0] - xd[0])**2 + (x1[1] - xd[1])**2
    d2_01 = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2

    if d2_0d > d2_1d + d2_01:
        return d2_1d
    if d2_1d > d2_0d + d2_01:
        return d2_0d

    return abs((x0[0] - x1[0]) * xd[1] - (x0[1] - x1[1]) * xd[0] - x1[1] * x0[0] + x1[0] * x0[1])\
        / math.sqrt(d2_01)

def huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))

def minimize(optimizer, loss, var_list):
    gradients = optimizer.compute_gradients(loss, var_list=var_list)
    return optimizer.apply_gradients(gradients)

