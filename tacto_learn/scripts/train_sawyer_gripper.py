import gym
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env

import tacto_learn.envs
from tacto_learn.envs.wrappers import SawyerActionWrapper, SawyerObservationWrapper
from tacto_learn.models.feature_extractors import CustomCombinedExtractor

def sawyer_wrappers(env):
    return SawyerObservationWrapper(SawyerActionWrapper(env))

def train_PPO():
    venv = make_vec_env(
        "sawyer-gripper-v0", 
        n_envs=4,
        wrapper_class=sawyer_wrappers,
    )
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    model = PPO(
        'MultiInputPolicy', 
        venv, 
        n_steps=256,
        batch_size=64, 
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("ppo_sawyer")   

def train_SAC():
    env = gym.make("sawyer-gripper-v0")
    env = SawyerObservationWrapper(SawyerActionWrapper(env))


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
    model.learn(total_timesteps=400, log_interval=2)
    model.save("sac_sawyer")   


if __name__ == '__main__':
    train_SAC()
