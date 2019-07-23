import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ''))

from utils import *

import tensorflow as tf
import numpy as np
import glob
from itertools import compress

slim = tf.contrib.slim


class Model:
    def __init__(self, hwc, ex_dim, action_dim, args):
        self.args = args
        selectGpuById(args.gpu)
        self.hwc = hwc
        self.h = hwc[0]
        self.w = hwc[1]
        self.c = hwc[2]
        self.ex_dim = ex_dim

        self.action_dim = action_dim

    def generate_sess(self):
        try:
            self.sess
        except AttributeError:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)

    def initialize(self):
        print("Model Initialization starts")
        self.generate_sess()

        '''uninitialized_variables = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_variables.append(var)
        self.sess.run(tf.variables_initializer(uninitialized_variables))'''

        '''uninitialized_variables = []
        for name in self.sess.run(tf.report_uninitialized_variables(tf.global_variables())):
            name = name.decode('utf-8')
            try:
                with slim.variable_scope('', reuse=True):  
                    uninitialized_variables.append(tf.get_variable(name))
            except ValueError:
                print("Failed to collect variable {}. Skipping.", name)
        self.sess.run(tf.variables_initializer(uninitialized_variables))'''
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
                                            for var in global_vars])
        uninitialized_variables = list(compress(global_vars, is_not_initialized))
        if len(uninitialized_variables):
            self.sess.run(tf.variables_initializer(uninitialized_variables))

        print("Variables initialized:")
        print(vars_info(uninitialized_variables))
        # self.sess.run(tf.global_variables_initializer())

        self.start_iter = 0
        print("Model Initialization ends")

    def save(self, global_step, save_dir):
        for f in glob.glob(save_dir + '*'): os.remove(f)
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess, os.path.join(save_dir, 'model'), global_step=global_step)
        print("Model save in %s" % save_dir)

    def restore(self, save_dir=None, checkpoint=None, reset_iter=False):
        print("Restoring model starts...")
        self.start_iter = 0
        saver = tf.train.Saver()
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(save_dir)
            self.start_iter = int(os.path.basename(checkpoint)[len('model') + 1:])
            print("Restoring from {}".format(checkpoint))
        if reset_iter == True:
            self.start_iter = 0
        self.generate_sess()
        saver.restore(self.sess, checkpoint)
        print("Restoring model done.")

    def regen_session(self):
        tf.reset_default_graph()
        self.sess.close()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
