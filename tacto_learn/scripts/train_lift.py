import os
import os.path as osp
import argparse
from ruamel.yaml import YAML

import robosuite as suite
from stable_baselines3 import SAC 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from tacto_learn.utils.wrappers import GymWrapper
from tacto_learn.utils.extractors import DictExtractor

def make_env(env_cfg=None):
    controller_configs = suite.load_controller_config(
        default_controller=env_cfg['controller_configs']['default_controller']
    )
    env_cfg['controller_configs'] = controller_configs
    env = suite.make(**env_cfg)
    env = GymWrapper(env)
    return env

def train_policy(env, policy_cfg):

    policy_kwargs = dict(
        features_extractor_class=DictExtractor,
    )

    policy_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../trained_models'))
    os.makedirs(policy_path, exist_ok=True)

    if policy_cfg['policy_name'] == 'SAC':
        policy_cls = SAC
    else:
        assert False, f"{policy_cfg['policy_name']} not implemented"
    policy = policy_cls(
        'MultiInputPolicy', 
        env,
        buffer_size=policy_cfg['buffer_size'], 
        policy_kwargs=policy_kwargs
    )

    from stable_baselines3.common.logger import configure
    data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../logs'))
    policy_name = policy_cfg['policy_name']
    tmp_path = data_path + f'/{policy_name}'

    # set up logger
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])
    policy.set_logger(new_logger)
    policy.learn(total_timesteps=policy_cfg['total_timesteps'], log_interval=4)

    policy_name = policy_path + f'/{policy_name}'
    policy.save(policy_name)
    return policy


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

    # make env
    env = make_vec_env(
        make_env, 
        env_kwargs=dict(env_cfg=cfg['env']), 
        vec_env_cls=SubprocVecEnv, 
        n_envs=8
    )
    # env = SubprocVecEnv([make_env(cfg['env']) for])
    # env = make_env(cfg['env'])

    # train policy
    policy = train_policy(env, cfg['policy'])

if __name__ == '__main__':
    main()