from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .encoders import ImageEncoder, VectorEncoder

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        img_lat_dim = 32
        vec_lat_dim = 8

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if 'color' in key or 'depth' in key:
                if 'depth' in key:
                    input_shape = (1, *subspace.shape)
                elif subspace.shape[0] <= 4:
                    input_shape = subspace.shape
                else:
                    input_shape = (subspace.shape[-1], *subspace.shape[:2])
                extractors[key] = ImageEncoder(input_shape, img_lat_dim)
                total_concat_size += img_lat_dim
            # elif "depth" in key:
            #     extractors[key] = DepthEncoder(subspace.shape, z_dim)
            else:
                assert len(subspace.shape) == 1, (key, subspace)
                extractors[key] = VectorEncoder(subspace.shape[0], vec_lat_dim)
                total_concat_size += vec_lat_dim

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            x = observations[key]
            if 'depth' in key:
                x = x.unsqueeze(dim=1)
            feature = extractor(x)
            encoded_tensor_list.append(feature)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
