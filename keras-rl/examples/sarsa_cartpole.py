#WandB init
from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np
import wandb

from keras import __version__ as KERAS_VERSION
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from keras.utils.generic_utils import Progbar

#cartpole init
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


#WandB classes
class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass
    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass

class WandbLogger(Callback):
    """ Similar to TrainEpisodeLogger, but sends data to Weights & Biases $ """

    def __init__(self, **kwargs):
        kwargs = {
            'project': 'keras-rl',
            'anonymous': 'allow',
            **kwargs
        }
        wandb.init(**kwargs)
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        wandb.config.update({
            'params': self.params,
            'env': self.env.__dict__,
            'env.env': self.env.env.__dict__,
            'env.env.spec': self.env.env.spec.__dict__,
            'agent': self.model.__dict__
        })

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and log training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        metrics = np.array(self.metrics[episode])
        metrics_dict = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    metrics_dict[name] = np.nanmean(metrics[:, idx])
                except Warning:
                    metrics_dict[name] = float('nan')

        wandb.log({
            'step': self.step,
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            **metrics_dict
        })

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

print(model.summary())

# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2, callbacks=[WandbLogger()])

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=5, visualize=True)
