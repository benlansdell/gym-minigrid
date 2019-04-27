#Don't use the GPU for these runs:
import os 
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
import tensorflow as tf
from a2c import get_actor_critic, CnnPolicy

from gym_minigrid.blocks import *
from gym_minigrid.register import register

#####################
# Task environments #
#####################

#These environments overwrite the step function to provide different dynamics and to modify
#information available to the learning algorithm (e.g. perhaps the agent isn't visible)

class BlockMazeEnv_Other(BlockMazeEnv):
    """
    Instead of listening to action, we just sample action from another already learnt policy...
    """

    def __init__(self):
        fn_policy = 'weights/a2c_400000.ckpt'
        tf.reset_default_graph()
        self.sess = tf.Session()
        nenvs = 1
        nsteps = 1
        ob_space = self.observation_space
        ac_space = self.action_space
        with tf.variable_scope('actor'):
            self.actor_critic = get_actor_critic(self.sess, nenvs, nsteps, ob_space,
                    ac_space, CnnPolicy, should_summary=False)
        self.actor_critic.load(fn_policy)
        super().__init__()

    def _run_other_policy(self):
        #prepare states...
        obs = self.gen_obs()
        actions, _, _ = self.actor_critic.act(np.expand_dims(obs, axis=0))
        return action

    #A wrapper for the default BlockMaze stepper we use to modify the dynamics of the environment
    def step(self, requested_action):
        action = self._run_other_policy()
        return self._step(action)

# Tasks
register(
    id='MiniGrid-BlockMaze-Other-v0',
    entry_point='gym_minigrid.envs:BlockMazeEnv_Other'
)
