from typing import Optional

import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3
        assert obs_shape[0] == 3, "Require channel first"

        self.feature_dim = feature_dim

        layers = [
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())

        self.convs = nn.Sequential(*layers)
        # Assumes input image is 84x84
        self.fc = nn.Linear(num_filters*39*39, self.feature_dim)

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
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
            if key == "agentview_image":
                extractors[key] = CNN(observation_space[key].shape, feature_dim=image_feature_dim)
                total_concat_size += image_feature_dim
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