import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'         # Offscreen rendering

import numpy as np
import gym
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env

import tacto_learn.envs
from tacto_learn.envs.wrappers import SawyerActionWrapper, SawyerObservationWrapper
from tacto_learn.models.bc import BC
from tacto_learn.models.policy import GraspingPolicy
from tacto_learn.models.feature_extractors import CustomCombinedExtractor
from tacto_learn.utils import utils


def sawyer_wrappers(env):
    return SawyerObservationWrapper(SawyerActionWrapper(env))

def train_BC():
    env = gym.make("sawyer-gripper-v0")
    env = sawyer_wrappers(env)

    expert_policy = GraspingPolicy(env)
    bc = BC(env, expert_policy)
    bc.train()


# def train_PPO():
#     venv = make_vec_env(
#         "sawyer-gripper-v0", 
#         n_envs=4,
#         wrapper_class=sawyer_wrappers,
#     )
    
#     policy_kwargs = dict(
#         features_extractor_class=CustomCombinedExtractor,
#     )

#     model = PPO(
#         'MultiInputPolicy', 
#         venv, 
#         n_steps=256,
#         batch_size=64, 
#         n_epochs=10,
#         policy_kwargs=policy_kwargs,
#         verbose=1
#     )
#     model.learn(total_timesteps=10000, log_interval=4)
#     model.save("ppo_sawyer")   

def train_SAC():
    env = gym.make("sawyer-gripper-v0")
    env = sawyer_wrappers(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    model = SAC(
        'MultiInputPolicy', 
        env,
        buffer_size=10000, 
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    model.learn(total_timesteps=100000, log_interval=2)
    model.save("sac_sawyer")   

    env.close()

if __name__ == '__main__':
    train_BC()
