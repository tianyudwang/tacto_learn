from typing import Dict
import abc

import numpy as np
import torch as th
from torch import nn
import gym
from stable_baselines3.common import policies

from .encoders import ImageEncoder, VectorEncoder
import tacto_learn.utils.pytorch_utils as ptu

class BasePolicy(nn.Module):
    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, ob):
        raise NotImplementedError

class GraspingPolicy(BasePolicy):
    """Hard-coded policy"""
    def __init__(self, env):
        super().__init__(env)
        self.t = 0

        self.z_low, self.z_high = 0.05, 0.4
        self.dz = 0.02
        self.w_open, self.w_close = 0.11, 0.05
        self.gripper_force = 20

    def reset(self):
        self.t = 0

    def predict(self, ob):
        action = np.zeros(self.env.action_space.shape)

        if self.t < 50:
            action[:3] = ob['object_position'] + np.array([0, 0, self.z_high])
            action[3:7] = np.array([0.0, 1, 0.0, 0.0])
            action[7:8] = self.w_open
        elif self.t < 100:
            s = (self.t - 50) / 50
            z = self.z_high - s * (self.z_high - self.z_low)
            action[:3] = ob['object_position'] + np.array([0, 0, z])
        elif self.t < 150:
            action[7:8] = self.w_close
            action[8:9] = self.gripper_force
        elif self.t < 220:
            delta = [0, 0, self.dz]
            action[:3] = ob['robot_end_effector_position'] + delta
            action[7:8] = self.w_close
            action[8:9] = self.gripper_force
        else:
            action[7:8] = self.w_close

        self.t += 1
        return action

# class MultiModalPolicy(BasePolicy):
#     def __init__(self, env):
#         super().__init__(env)

#         self.observation_space = self.env.observation_space
#         self.action_space = self.env.action_space

#         img_lat_dim = 32
#         vec_lat_dim = 8
        
#         extractors = {}
#         total_concat_size = 0
#         for mode, subspace in self.observation_space.spaces.items()
#             if 'color' in mode or 'depth' in mode:
#                 if 'depth' in key:
#                     input_shape = (1, *subspace.shape)
#                 else:
#                     input_shape = subspace.shape
#                 extractors[mode] = ImageEncoder(input_shape, img_lat_dim)
#                 total_concat_size += img_lat_dim
#             else:
#                 assert len(subspace.shape) == 1, (mode, subspace)
#                 extractors[key] = VectorEncoder(subspace.shape[0], vec_lat_dim)
#                 total_concat_size += vec_lat_dim
#         self.extractors = nn.ModuleDict(extractors)
#         self.features_dim = total_concat_size

#         self.mlp = ptu.build_mlp(
#             input_size=self.features_dim,
#             size=128,
#             n_layers=2,
#             activation=2,
#             output_size=self.action_space.shape[0],
#             output_activation='identity'
#         )

#     def reset(self):
#         pass

#     def forward(self, obs: Dict[str, th.Tensor]):
#         encoded_tensor_list = []
#         for mode, extractor in self.extractors.items():
#             x = obs[mode]
#             feature = extractor(x)
#             encoded_tensor_list.append(feature)
#         x = th.cat(encoded_tensor_list, dim=1)
#         x = self.mlp(x)
#         return x

#     def predict(self, ob):
#         new_ob = {}
#         for mode, val in ob.items():
#             new_ob[mode] = ptu.from_numpy(val).unsqueeze(0)
#         with torch.no_grad():
#             act = self(new_ob).squeeze(0)
#         act = ptu.to_numpy(act)
#         return act

class MultiModalPolicy(policies.ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs)

    def predict(
        self, 
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> np.ndarray:
        new_observation = {}
        for mode, val in observation.items():
            if isinstance(val, float):
                val = np.array([val]).reshape(1,1)
            else:
                val = val[None, ...]
            val = ptu.from_numpy(val)
            if 'color' in mode:
                val = val.permute(0,3,1,2)
            elif 'depth' in mode:
                val = val.unsqueeze(dim=1)
            new_observation[mode] = val

        with th.no_grad():
            actions = self._predict(new_observation, deterministic=deterministic)

        actions = np.squeeze(ptu.to_numpy(actions))
        if isinstance(self.action_space, gym.spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions