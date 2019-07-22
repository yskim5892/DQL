import tensorflow as tf
from Model import Model
import numpy as np

class DQNetwork(Model):
    def __init__(self, state_shape, action_dim, args):
        Model.__init__(state_shape, action_dim, args)

    def build(self):
        print("Model building starts")
        tf.reset_default_graph()
        self.generate_sess()

        self.state = tf.placeholder(tf.float32, shape = \
            [None, self.state_shape[0], self.state_shape[1], self.state_shape[2]])
        self.Q = self.Q_net(self.state)


        self.action = tf.placeholder(tf.float32, shape = [None, self.action_dim])
        self.reward = tf.placeholder(tf.float32, shape = [None, ])
        self.next_state = tf.placeholder(tf.float32, shape = \
            [None, self.state_shape[0], self.state_shape[1], self.state_shape[2]])

        Q_next = self.Q_net(self.next_state)
        goal = self.reward + self.args.gamma * tf.reduce_max(Q_next, 1)
        current_Q = tf.reduct_sum(tf.mul(self.Q, self.action), 1)

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
        layer = state
        layer = tf.layers.conv2d(layer, 64, [3, 3], activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 32, [3, 3], activation=tf.nn.relu)
        layer = tf.reshape(layer, [-1])
        return tf.contrib.layers.fully_connected(layer, self.action_dim, activation=tf.nn.softmax)

    def Q_value(self, state):
        return self.sess.run(self.Q, feed_dict={self.state : state})

    def learn_from_history(self, state_history, action_history, reward_history, next_state_history):
        step = self.sess.run(self.global_step)
        print("step : ", step)
        if step >= self.args.max_experience:
            indices = np.random.choice(self.args.max_experience, self.args.batch_size)
            feed_dict = {self.state : state_history[indices], self.action : action_history[indices],\
                         self.reward : reward_history[indices], self.next_state : next_state_history[indices]}
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            print("loss : ", loss)

