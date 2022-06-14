import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'         # Offscreen rendering

import numpy as np
import gym
import time
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env ,SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import tacto_learn.envs
from tacto_learn.envs.wrappers import SawyerActionWrapper, SawyerObservationWrapper, SawyerStateObservationWrapper, SawyerVecStateObservationWrapper
from tacto_learn.models.bc import BC
from tacto_learn.models.policy import GraspingPolicy, GraspingPolicy2
from tacto_learn.models.feature_extractors import CustomCombinedExtractor
from tacto_learn.utils import utils
import torch as th
th.autograd.set_detect_anomaly(True)

def sawyer_wrappers(env):
    # return SawyerStateObservationWrapper(SawyerActionWrapper(env))
    return SawyerVecStateObservationWrapper(SawyerActionWrapper(env))
    # return SawyerObservationWrapper(SawyerActionWrapper(env))


def make_env():
    env = gym.make("sawyer-gripper-v1")
    env = sawyer_wrappers(env)
    # env.num_envs=8
    # env = VecCheckNan(env, raise_exception=True)
    return env
    

def train_SAC():
    venv = make_vec_env(make_env,
    vec_env_cls=SubprocVecEnv,n_envs=8)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    # env = gym.make("sawyer-gripper-v1")
    # env = sawyer_wrappers(env)
    # env = VecCheckNan(env, raise_exception=True)

    model = SAC(
        'MultiInputPolicy', 
        venv,
        buffer_size=10000, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="/home/siddhant/tacto_ws/tensorboard_logs/SAC/StateVec/pick",gradient_steps=4
    )

    model.learn(total_timesteps=2000000, log_interval=2)
    model.save("sac_StateVec_2M_reach")

    venv.close()
def train_PPO():
    venv = make_vec_env(make_env,
    vec_env_cls=SubprocVecEnv,n_envs=16)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    # env = gym.make("sawyer-gripper-v1")
    # env = sawyer_wrappers(env)

    model = PPO(
        'MultiInputPolicy', 
        venv, 
        n_steps=256//16,
        batch_size=64, 
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="/home/siddhant/tacto_ws/tensorboard_logs/PPO/stateVec/reach_pick"
    )
    # model = PPO.load("/home/siddhant/tacto_ws/ppo_sawyer_full_obs_fixed_hirozon_3M_d",tensorboard_log="/home/siddhant/tacto_ws/tensorboard_logs/PPO/full_obs/fixed_horizon_d")
    # model.set_env(venv)


    model.learn(total_timesteps=3000000, log_interval=2)
    model.save("ppo_sawyer_full_obs_fixed_hirozon_reach_pick")

    venv.close()


def train_BC():
    env = gym.make("sawyer-gripper-v1")
    env = sawyer_wrappers(env)

    # expert_policy = GraspingPolicy(env)
    expert_policy = GraspingPolicy2(env,PPO.load("/home/siddhant/tacto_ws/ppo_sawyer_state_fixed_hirozon_3M_c9",env=env))
    bc = BC(env, expert_policy)
    bc.train()


if __name__ == '__main__':
    t1 = time.time()

    train_SAC()
    # train_PPO()
    # train_BC()
    t2= time.time()
    print("Done",t2-t1)
