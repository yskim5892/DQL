import tensorflow as tf
from Model import Model
import numpy as np

class DQNetwork(Model):
    def __init__(self, hwc, ex_dim, action_dim, args):
        Model.__init__(self, hwc, ex_dim, action_dim, args)

    def build(self):
        print("Model building starts")
        tf.reset_default_graph()
        self.generate_sess()

        self.state = tf.placeholder(tf.float32, shape = \
                [None, self.h * self.w * self.c + self.ex_dim])

        self.Q = self.Q_net(self.state)

        self.action = tf.placeholder(tf.float32, shape = [None, self.action_dim])
        self.reward = tf.placeholder(tf.float32, shape = [None, ])

        self.next_state = tf.placeholder(tf.float32, shape = \
                [None, self.h * self.w * self.c + self.ex_dim])

        Q_next = self.Q_net(self.next_state)
        goal = self.reward + self.args.gamma * tf.reduce_max(Q_next, 1)
        current_Q = tf.reduce_sum(self.Q * self.action, 1)

        self.loss = tf.reduce_mean(tf.square(goal - current_Q))
        print("Model building ends")

    def build_train_op(self):
        print("Model building trian operation starts")
        self.global_step = tf.Variable(0, trainable=False)
        update_step_op = tf.assign_add(self.global_step, 1)

        lr = tf.train.exponential_decay(self.args.lr, self.global_step, self.args.decay_steps, self.args.decay_rate, staircase=True)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        print("Model building trian operation ends")

    def Q_net(self, state):
        image = tf.slice(state, [0, 0], [-1, self.h * self.w * self.c])
        image = tf.reshape(image, [-1, self.h, self.w, self.c])
        ex = tf.slice(state, [0, self.h * self.w * self.c], [-1, self.ex_dim])

        layer = image
        layer = tf.layers.conv2d(layer, 64, [3, 3], padding='SAME', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 32, [3, 3], padding='SAME', activation=tf.nn.relu)
        layer = tf.reshape(layer, [-1, self.h * self.w * 32])
        
        layer = tf.concat([layer, ex], 1)
        layer = tf.contrib.layers.fully_connected(layer, self.h * self.w * self.c + self.ex_dim)
        layer = tf.contrib.layers.fully_connected(layer, self.action_dim)
        return tf.nn.softmax(layer, axis=1)

        '''layer = state
        layer = tf.contrib.layers.fully_connected(layer, self.h * self.w * self.c + self.ex_dim)
        layer = tf.contrib.layers.fully_connected(layer, self.h * self.w * self.c + self.ex_dim)
        layer = tf.contrib.layers.fully_connected(layer, self.action_dim)
        return tf.nn.softmax(layer, axis=1)'''

    def Q_value(self, state):
        return self.sess.run(self.Q, feed_dict={self.state : [state]})

    def learn_from_history(self, state_history, action_history, reward_history, next_state_history):
        if len(state_history) >= self.args.max_experience:
            indices = np.random.choice(self.args.max_experience, self.args.batch_size)
            states = [state_history[i] for i in indices]
            actions = [action_history[i] for i in indices]
            rewards = [reward_history[i] for i in indices]
            next_states = [next_state_history[i] for i in indices]

            feed_dict = {self.state : states, self.action : actions,\
                         self.reward : rewards, self.next_state : next_states}
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            return loss
        return None

