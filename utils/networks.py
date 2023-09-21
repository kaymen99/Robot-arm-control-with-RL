import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, name, model, checkpoints_dir="ckp/"):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir + model, name + ".h5")

        self.layer1 = Dense(512, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.layer3 = Dense(256, activation="relu")
        self.pi = Dense(n_actions, activation="tanh")
    
    @tf.function()
    def call(self, state):
        action = self.layer1(state)
        action = self.layer2(action)
        action = self.layer3(action)
        pi = self.pi(action)

        return pi

class CriticNetwork(keras.Model):
    def __init__(self, name, model, checkpoints_dir="ckp/"):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir + model, name + ".h5")

        self.layer1 = Dense(512, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.layer3 = Dense(256, activation="relu")
        self.q = Dense(1, activation=None)

    @tf.function()
    def call(self, state, action):
        value = self.layer1(tf.concat([state, action], axis=1))
        value = self.layer2(value)
        value = self.layer3(value)
        q = self.q(value)

        return q
