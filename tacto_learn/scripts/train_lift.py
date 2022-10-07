import os
import os.path as osp
import argparse
from ruamel.yaml import YAML

import robosuite as suite
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from tacto_learn.utils.wrappers import GymWrapper
# from robosuite.wrappers import GymWrapper
from tacto_learn.utils.extractors import DictExtractor

def make_env(env_cfg=None):
    controller_configs = suite.load_controller_config(
        default_controller=env_cfg['controller_configs']['default_controller']
    )
    env_cfg['controller_configs'] = controller_configs

    # Omit object true state if using camera observation
    if env_cfg['use_camera_obs']:
        env_cfg['use_object_obs'] = False
    else:
        env_cfg['use_object_obs'] = True
 
    env = suite.make(**env_cfg)
    env = GymWrapper(env)
    return env

def train_policy(env, policy_cfg, exp_name):

    # For SAC and DDPG, feature extractors should not be shared between actor and critic
    # See https://github.com/DLR-RM/stable-baselines3/pull/935
    policy_kwargs = dict(
        features_extractor_class=DictExtractor,
        features_extractor_kwargs=policy_cfg['features_extractor'],
        share_features_extractor=False,
    )

    policy_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../trained_models'))
    os.makedirs(policy_path, exist_ok=True)

    policy_classes = {'SAC': SAC, 'DDPG': DDPG}
    policy_name = policy_cfg['policy_name']
    if policy_name in policy_classes:
        policy_cls = policy_classes[policy_name]
    else:
        assert False, f"{policy_name} not implemented"

    policy = policy_cls(
        'MultiInputPolicy', 
        env,
        buffer_size=policy_cfg['buffer_size'], 
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
    )

    from stable_baselines3.common.logger import configure
    log_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../logs'))
    tmp_path = osp.join(log_path, exp_name)

    # set up logger
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])
    policy.set_logger(new_logger)
    policy.learn(total_timesteps=policy_cfg['total_timesteps'], log_interval=4)

    policy_name = osp.join(policy_path, exp_name)
    policy.save(policy_name)
    return polic


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visual tactile RL training in robosuite')
    parser.add_argument('config', help='train config file path')

    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    cfg = yaml.load(open(args.config))

    # Merge command line arguments with configs
    if args.seed is not None:
        cfg['seed'] = args.seed
    print(cfg)

    # Generate experiment name 
    # algorithm + env_name + robots + controller + obs_mode
    env_cfg, policy_cfg = cfg['env'], cfg['policy']
    
    obs_mode = ['proprio']
    if env_cfg['use_camera_obs']:
        obs_mode.append('visual')
    else:
        obs_mode.append('obj_state')
    if env_cfg['use_touch_obs']:
        obs_mode.append('touch')
    obs_mode = '_'.join(obs_mode)

    exp_name_args = [
        policy_cfg['policy_name'],
        env_cfg['env_name'],
        env_cfg['robots'],
        env_cfg['controller_configs']['default_controller'],
        obs_mode,
    ]
    exp_name = '_'.join(exp_name_args)

    if policy_cfg.get("suffix", None):
        exp_name += "_" + policy_cfg["suffix"]

    # make env
    env = make_vec_env(
        make_env, 
        env_kwargs=dict(env_cfg=env_cfg), 
        vec_env_cls=SubprocVecEnv, 
        n_envs=8
    )
    # env = make_env(env_cfg)

    # train policy
    policy = train_policy(env, policy_cfg, exp_name)

if __name__ == '__main__':
    main()