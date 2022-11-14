import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as p

import utils


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class Encoder(nn.Module):
    def __init__(self, obs_shape, spectral_norm=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.feat_dim = 50

        # self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU())

        convs = [
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.Conv2d(32, 32, 3, stride=1),
        ]
        layers = []
        for conv in convs:
            if spectral_norm:
                layer = p.spectral_norm(conv)
            else:
                layer = conv 
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))
        self.convnet = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(self.repr_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim), 
            nn.Tanh()
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        return h

class MLP(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        n_layers=2, 
        size=256, 
        activation='relu',
        output_activation='identity',
        spectral_norm=False
    ):
        super().__init__()

        self.feat_dim = output_size
    
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layer = nn.Linear(in_size, size)
            if spectral_norm:
                layer = p.spectral_norm(layer)
            layers.append(layer)
            layers.append(activation)
            in_size = size
        layer = nn.Linear(in_size, output_size)
        if spectral_norm:
            layer = p.spectral_norm(layer)
        layers.append(layer)
        layers.append(output_activation)

        self.mlp = nn.Sequential(*layers)

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.mlp(x)


class FeatureExtractor(nn.Module):
    def __init__(self, obs_shape, spectral_norm=False):
        super().__init__()        

        self.encoders = nn.ModuleDict()
        for k, shape in obs_shape.items():
            if k == 'agentview_image' or k == 'observation':
                # Multiply channel with frame_stacks
                self.encoders[k] = Encoder(shape, spectral_norm=spectral_norm)
                self.image_key = k
            elif k == 'robot0_proprio-state':
                self.encoders[k] = MLP(shape[0], 64, spectral_norm=spectral_norm)
            elif k == 'object-state':
                self.encoders[k] = MLP(shape[0], 64, spectral_norm=spectral_norm)
            elif k =='robot0_touch-state':
                self.encoders[k] = MLP(shape[0], 32, spectral_norm=spectral_norm)
            else:
                raise ValueError(f'Encoder for {k} not implemented')

        self.feat_dim = sum([encoder.feat_dim for encoder in self.encoders.values()])


    def forward(self, obs):
        assert isinstance(obs, dict), "Observation must be wrapped in dictionary"
        obs_feats = []
        for k, ob in obs.items():
            assert len(ob.shape) == 2 or len(ob.shape) == 4
            obs_feats.append(self.encoders[k](ob))
        obs_feat = torch.cat(obs_feats, dim=-1)
        return obs_feat