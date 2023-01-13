from typing import Optional

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()
        print("obs shape", obs_shape, "feature dim", feature_dim)

        assert len(obs_shape) == 3
        assert obs_shape[0] == 3 or obs_shape[0] == 2, f"Require channel first, obs_shape {obs_shape}"
        self.obs_shape = obs_shape

        layers = [
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())

        self.convs = nn.Sequential(*layers)

        out_shape = self._compute_conv_out_shape()
        # self.repr_dim = np.prod(out_shape)
        self.fc = nn.Linear(np.prod(out_shape), feature_dim)

    def _compute_conv_out_shape(self):
        x = th.rand((1,) + self.obs_shape)
        return self.convs(x).shape

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

    def forward(self, obs):
        # obs = obs / 255.0 - 0.5
        # import ipdb; ipdb.set_trace()
        obs -= 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class CNN1(nn.Module):
    def __init__(self, obs_shape):

        # obs_shape = (3, 84, 84)

        self.repr_dim = 32 * 13 * 13

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 8, kernel_size=3, stride=2),  # resultant shape should be (8, 41, 41)
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=3, stride=1),  # (32, 39, 39)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)       # (32, 13, 13)
        )

    def forward(self, obs):
        x = self.convnet(obs)
        x = x.view(x.shape[0], -1)
        return x


class DictExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: gym.spaces.Dict, 
        image_feature_dim: Optional[int] = -1,
        proprio_feature_dim: Optional[int] = -1,
        object_feature_dim: Optional[int] = -1,
        touch_feature_dim: Optional[int] = -1,
        hidden_dim: Optional[int] = 256,
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(DictExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            print("we are processing key", key)
            if "image" in key or "tactile_depth" in key:
                # extractors[key] = CNN(subspace.shape, image_feature_dim)
                extractors[key] = CNN1(subspace.shape)
                # extractors[key] = Encoder(subspace.shape)
                total_concat_size += extractors[key].repr_dim
            else:
                # Run through a simple MLP
                if "proprio" in key:
                    feature_dim = proprio_feature_dim               
                elif "touch" in key:
                    feature_dim = touch_feature_dim
                elif "object" in key:
                    feature_dim = object_feature_dim
                assert feature_dim > 0, f"Feature dimension should be provided for observation {key}"

                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, feature_dim),
                    nn.ReLU(),                
                )
                total_concat_size += feature_dim

                # extractors[key] = nn.Flatten()
                # total_concat_size += observation_space[key].shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)