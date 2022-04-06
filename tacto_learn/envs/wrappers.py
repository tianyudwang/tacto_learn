from collections import OrderedDict
import numpy as np
import gym
import pybulletX as px

def flatten_dict_space(space, flatten):
    if isinstance(space, px.utils.SpaceDict):
        for key, val in space.items():
            flatten_dict_space(val, flatten)
    else:
        assert isinstance(space, gym.spaces.Box), type(space)
        flatten['dim'] += space.shape[0]
        flatten['low'].append(space.low)
        flatten['high'].append(space.high)

def fill_dict_val(action, values):
    pass

# Merge sawyer action into one 
class SawyerActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.unwrapped_action_space = self.unwrapped.action_space
        
        flatten = {'dim': 0, 'low': [], 'high': []}
        flatten_dict_space(self.unwrapped_action_space, flatten)
        self.action_space = gym.spaces.Box(
            low=np.concatenate(flatten['low']),
            high=np.concatenate(flatten['high']),
            shape=(flatten['dim'],),
            dtype=np.float32
        )

    def action(self, act):
        # Converts numpy action into original dict
        new_act = self.unwrapped_action_space.new()
        new_act.end_effector.position = act[0:3]
        new_act.end_effector.orientation = act[3:7]
        new_act.gripper_width = act[7:8]
        new_act.gripper_force = act[8:9]
        return new_act

# Convert nested dict into flatten dict
class SawyerObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.unwrapped_observation_space = self.unwrapped.observation_space

        observation_space = {
            'camera_color': self.unwrapped_observation_space['camera']['color'],
            'camera_depth': self.unwrapped_observation_space['camera']['depth'],
            'digits_color_0': self.unwrapped_observation_space['digits'][0]['color'],
            'digits_depth_0': self.unwrapped_observation_space['digits'][0]['depth'],
            'digits_color_1': self.unwrapped_observation_space['digits'][1]['color'],
            'digits_depth_1': self.unwrapped_observation_space['digits'][1]['depth'],
            'robot_end_effector_position': self.unwrapped_observation_space['robot']['end_effector']['position'],
            'robot_end_effector_orientation': self.unwrapped_observation_space['robot']['end_effector']['orientation'],
            'robot_gripper_width': self.unwrapped_observation_space['robot']['gripper_width'],
            'object_position': self.unwrapped_observation_space['object']['position'],
            'object_orientiation': self.unwrapped_observation_space['object']['orientation'],
        }
        
        self.observation_space = gym.spaces.Dict(observation_space)

    def observation(self, obs):
        # modify obs
        new_obs = {
            'camera_color': obs['camera']['color'],
            'camera_depth': obs['camera']['depth'],
            'digits_color_0': obs['digits'][0]['color'],
            'digits_depth_0': obs['digits'][0]['depth'],
            'digits_color_1': obs['digits'][1]['color'],
            'digits_depth_1': obs['digits'][1]['depth'],
            'robot_end_effector_position': obs['robot']['end_effector']['position'],
            'robot_end_effector_orientation': obs['robot']['end_effector']['orientation'],
            'robot_gripper_width': obs['robot']['gripper_width'],
            'object_position': obs['object']['position'],
            'object_orientiation': obs['object']['orientation']
        }
        return new_obs