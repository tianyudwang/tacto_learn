import os
import os.path as osp
import argparse
from ruamel.yaml import YAML

from stable_baselines3 import SAC
import robosuite as suite
from tacto_learn.utils.wrappers import GymWrapper

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


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualizing trained policy in robosuite')
    parser.add_argument('config', help='train config file path')

    parser.add_argument('--trained_model', type=str, default=None, help='Path to trained model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    cfg = yaml.load(open(args.config))
    env_cfg, policy_cfg = cfg['env'], cfg['policy']

    # Turn on renderer
    env_cfg['has_renderer'] = True
    env_cfg['has_offscreen_renderer'] = False

    env = make_env(env_cfg)
    obs = env.reset()

    policy = SAC.load(args.trained_model)

    touch_sensor_names = ['gripper0_touch1', 'gripper0_touch2']

    # while not done:
    for i in range(1000):
        action, _ = policy.predict(obs) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display

        # ee_force = np.linalg.norm(env.robots[0].ee_force)
        # total_force_ee = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))

        # touch_sensor_values = [env.robots[0].get_sensor_measurement(name)[0] for name in touch_sensor_names]

        # print(i, obs[-1], f"{ee_force:.3f}", f"{total_force_ee:.3f}")
        # print(f"Step {i}, contact {obs[-3]}, pressure {obs[-2:]}")


if __name__ == '__main__':
    main()