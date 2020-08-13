import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines Pei-Chen Wu ################ 

        with tf.variable_scope(scope, reuse = reuse) as _:
            conv1 = layers.conv2d(inputs = state, num_outputs = 32, kernel_size = 8, stride = 4, padding = 'same')
            conv2 = layers.conv2d(inputs = conv1, num_outputs = 64, kernel_size = 4, stride = 2, padding = 'same')
            conv3 = layers.conv2d(inputs = conv2, num_outputs = 64, kernel_size = 3, stride = 1, padding = 'same')
            fully_connected_layer = layers.fully_connected(inputs = layers.flatten(conv3), num_outputs = 512)
            out = layers.fully_connected(inputs = fully_connected_layer, num_outputs = num_actions, activation_fn = None)

        #with tf.variable_scope(scope, reuse = reuse) as _:
        #    conv1 = tf.layers.conv2d(inputs = state, filters = 32, kernel_size = (8,8), strides = 4)
        #    conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = (4,4), strides = 2)
        #    conv3 = tf.layers.conv2d(inputs = conv2, filters = 64, kernel_size = (3,3), strides = 1)
        #    fully_connected_layer = layers.fully_connected(inputs = tf.layers.flatten(conv3), num_outputs = 512, activation_fn = tf.nn.relu)
        #    out = layers.fully_connected(inputs = fully_connected_layer, num_outputs = num_actions, activation_fn = None)


        #with tf.variable_scope(scope, reuse = reuse) as _:
        #    conv1 = tf.layers.conv2d(inputs = state, filters = 32, kernel_size = (8,8), strides = 4, activation = tf.nn.relu)
        #    conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = (4,4), strides = 2, activation = tf.nn.relu)
        #    conv3 = tf.layers.conv2d(inputs = conv2, filters = 64, kernel_size = (3,3), strides = 1, activation = tf.nn.relu)
        #    fully_connected_layer = layers.fully_connected(inputs = tf.layers.flatten(conv3), num_outputs = 512, activation_fn = tf.nn.relu)
        #    out = layers.fully_connected(inputs = fully_connected_layer, num_outputs = num_actions)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
