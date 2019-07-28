import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from Model import Model
import numpy as np
import utils

class DQNetwork(Model):
    def __init__(self, hwc, ex_dim, action_dim, args):
        Model.__init__(self, hwc, ex_dim, action_dim, args)

    def build(self):
        self.q_vars_dict = dict()
        self.target_q_vars_dict = dict()

        print("Model building starts")
        tf.reset_default_graph()
        self.generate_sess()

        self.state = tf.placeholder(tf.float32, shape = \
                [None, self.h * self.w * self.c + self.ex_dim])

        self.Q = self.Q_net(self.state, scope='q_func', trainable=True)

        self.action = tf.placeholder(tf.float32, shape = [None, self.action_dim])
        self.reward = tf.placeholder(tf.float32, shape = [None, ])

        self.next_state = tf.placeholder(tf.float32, shape = \
                [None, self.h * self.w * self.c + self.ex_dim])

        Q_next = self.Q_net(self.next_state, scope='target_q_func', trainable=False)
        is_terminal = tf.slice(self.next_state, [0, self.h * self.w * self.c + self.ex_dim - 1], [-1, 1]) # b * 1

        goal = self.reward + self.args.gamma * (1 - is_terminal) * tf.reduce_max(Q_next, 1)
        current_Q = tf.reduce_sum(self.Q * self.action, 1)

        self.loss = tf.reduce_mean(utils.huber_loss(goal - current_Q))

        self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        self.target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')
        update_target_fn = []
        for var, var_target in zip(sorted(self.q_func_vars, key=lambda v: v.name),
                                    sorted(self.target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        print("Model building ends")

    def build_train_op(self):
        print("Model building trian operation starts")
        self.global_step = tf.Variable(0, trainable=False)
        self.update_step_op = tf.assign_add(self.global_step, 1)

        self.lr = tf.train.exponential_decay(self.args.lr, self.global_step, self.args.decay_steps, self.args.decay_rate, staircase=True)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        #self.train_op = utils.minimize(tf.train.AdamOptimizer(self.lr), self.loss, var_list=self.q_func_vars)
        print("Model building trian operation ends")

    def Q_net(self, state, scope, trainable):
        with tf.variable_scope(scope, reuse=False):
            with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected], activation_fn=tf.nn.relu, trainable=trainable):
                image_ = tf.slice(state, [0, 0], [-1, self.h * self.w * self.c])
                image = tf.reshape(image_, [-1, self.h, self.w, self.c])
                ex = tf.slice(state, [0, self.h * self.w * self.c], [-1, self.ex_dim]) # b * ex_dim
                
                ### BHB-specific
                if self.args.task == 'BHB':
                    '''cur_block = tf.slice(ex, [0, 0], [-1, 27])   # b * 27
                    ex = tf.slice(ex, [0, 27], [-1, 2])       # b * 2
                    cur_block_tiled_to_every_pixel = tf.reshape(tf.tile(cur_block, [1, self.h * self.w]), [-1, self.h, self.w, 27]) # b * h * w * 27
                    image = tf.concat([image, cur_block_tiled_to_every_pixel], 3) # b * h * w * 61'''

                    layer = image
                    layer = tf.contrib.slim.conv2d(layer, 36, [5, 5], padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 36, [5, 5], padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 18, [5, 5], padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 18, [5, 5], padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 12, [5, 5], padding='SAME')

                    layer = tf.reshape(layer, [-1, self.h * self.w * 12])
                
                    layer = tf.concat([layer, ex], 1)
                    layer = tf.contrib.slim.fully_connected(layer, self.h * self.w * 12)
                    layer = tf.contrib.slim.fully_connected(layer, self.h * self.w * 6)
                    layer = tf.contrib.slim.fully_connected(layer, self.h * self.w * 3)
                    layer = tf.contrib.slim.fully_connected(layer, self.h * self.w)
                    layer = tf.contrib.slim.fully_connected(layer, self.h * self.w // 2)
                    layer = tf.contrib.slim.fully_connected(layer, self.h * self.w // 4)
                    layer = tf.contrib.slim.fully_connected(layer, self.action_dim, activation_fn = None)
                    return layer
                elif self.args.task == 'RT':
                    layer = image
                    # filters, kernel_size, strides
                    layer = tf.contrib.slim.conv2d(layer, self.c, 3, padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 2*self.c, 3, 2, padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 2*self.c, 3,    padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 4*self.c, 3, 2, padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 4*self.c, 3,    padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 8*self.c, 3, 2, padding='SAME')
                    layer = tf.contrib.slim.conv2d(layer, 8*self.c, 3,    padding='SAME')

                    dim = (self.h // 8) * (self.w // 8) * 8 * self.c
                    layer = tf.reshape(layer, [-1, dim])

                    layer = tf.concat([layer, ex, image_], 1)
                    layer = tf.contrib.slim.fully_connected(layer, 256)
                    layer = tf.contrib.slim.fully_connected(layer, 128)
                    layer = tf.contrib.slim.fully_connected(layer, 64)
                    layer = tf.contrib.slim.fully_connected(layer, 32)
                    layer = tf.contrib.slim.fully_connected(layer, self.action_dim, activation_fn = None)
                    return layer
        '''layer = state
        layer = tf.contrib.layers.fully_connected(layer, self.h * self.w * self.c + self.ex_dim)
        layer = tf.contrib.layers.fully_connected(layer, self.h * self.w * self.c + self.ex_dim)
        layer = tf.contrib.layers.fully_connected(layer, self.action_dim)
        return tf.nn.softmax(layer, axis=1)'''

    def Q_value(self, state):
        return self.sess.run(self.Q, feed_dict={self.state : [state]})

    def learn_from_history(self, state_history, action_history, reward_history, next_state_history):
        if len(state_history) >= self.args.batch_size * 2:
            indices = np.random.choice(len(state_history), self.args.batch_size)
            states = [state_history[i] for i in indices]
            actions = [action_history[i] for i in indices]
            rewards = [reward_history[i] for i in indices]
            next_states = [next_state_history[i] for i in indices]

            step = self.sess.run(self.global_step)
            if(step % self.args.target_update_period == 0):
                self.sess.run(self.update_target_fn)
            
            feed_dict = {self.state : states, self.action : actions,\
                         self.reward : rewards, self.next_state : next_states}
            loss, _, _ = self.sess.run([self.loss, self.train_op, self.update_step_op], feed_dict=feed_dict)

           
            # norm / difference check
            if(step % 1000 == 0):
                l2_dist = 0
                l2_norm = 0
                for var in self.q_func_vars:
                    value = self.sess.run(var)
                    if var in self.q_vars_dict:
                        l2_dist += np.sum(np.square(value - self.q_vars_dict[var]))
                        l2_norm += np.sum(np.square(value))
                    self.q_vars_dict[var] = value
                
                target_l2_dist = 0
                target_l2_norm = 0
                for var in self.target_q_func_vars:
                    value = self.sess.run(var)
                    if var in self.target_q_vars_dict:
                        target_l2_dist += np.sum(np.square(value - self.target_q_vars_dict[var]))
                        target_l2_norm += np.sum(np.square(value))
                    self.target_q_vars_dict[var] = value
                Q_norm = np.sqrt(np.sum(np.square(self.sess.run(self.Q, feed_dict=feed_dict))))
                print(step, loss, np.sqrt(l2_dist), np.sqrt(l2_norm), np.sqrt(target_l2_dist), np.sqrt(target_l2_norm), Q_norm)

            return loss
        return None

