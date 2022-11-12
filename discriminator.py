import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.utils.parametrizations as p

import utils

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.feat_dim = 50

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
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
    def __init__(self, in_dim, out_dim=128):
        super().__init__()

        self.feat_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.mlp(x)

class Discriminator:
    def __init__(
        self, 
        obs_shape,
        action_shape,
        device, 
        lr, 
        batch_size,
        hidden_dim,
        max_logit,
        reward_type
        # clip_reward_range: Optional[float] = -1.0,
    ):
        # super().__init__()

        self.device = device
        self.reward_type = reward_type
        self.max_logit = max_logit
        self.batch_size = batch_size
        # self.max_grad_norm = max_grad_norm

        # construct observation encoders based on observation spec
        self.encoders = {}
        self.image_key = None
        for k, shape in obs_shape.items():
            if k == 'agentview_image' or k == 'observation':
                # Multiply channel witorch frame_stacks
                self.encoders[k] = Encoder(shape).to(device)
                self.image_key = k
            elif k == 'robot0_proprio-state':
                self.encoders[k] = MLP(shape[0], out_dim=64).to(device)
            elif k == 'object-state':
                self.encoders[k] = MLP(shape[0], out_dim=64).to(device)
            elif k =='robot0_touch-state':
                self.encoders[k] = MLP(shape[0], out_dim=32).to(device)
            else:
                raise ValueError(f'Encoder for {k} not implemented')

        encoder_feat_dim = sum([encoder.feat_dim for encoder in self.encoders.values()])

        # models
        self.model = nn.Sequential(
            p.spectral_norm(nn.Linear(encoder_feat_dim+action_shape[0], hidden_dim)),
            nn.ReLU(inplace=True),
            p.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(inplace=True),
            p.spectral_norm(nn.Linear(hidden_dim, 1)),
        ).to(device)

        # optimizers
        self.encoder_opts = {}
        for k, v in self.encoders.items():
            self.encoder_opts[k] = torch.optim.Adam(self.encoders[k].parameters(), lr=lr)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.sigmoid = nn.Sigmoid()
        # self.softplus = nn.Softplus()
        self.loss = nn.BCELoss()
        
        # D = 0 for expert and D = 1 for agent
        demo_labels = torch.zeros((batch_size, 1), device=self.device)
        agent_labels = torch.ones((batch_size, 1), device=self.device) 
        self.labels = torch.cat((demo_labels, agent_labels), dim=0)

    def encode_obs(self, obs):
        assert isinstance(obs, dict), "Observation must be wrapped in dictionary"
        obs_feats = []
        for k, v in obs.items():
            ob = torch.as_tensor(v, device=self.device)
            # Add trivial batch dim during inference
            if len(ob.shape) == 1 or len(ob.shape) == 3:
                ob = ob.unsqueeze(0)
            obs_feats.append(self.encoders[k](ob))
        obs_feat = torch.cat(obs_feats, dim=-1)
        return obs_feat

    def forward(self, obs, act):
        obs = self.encode_obs(obs)
        logits = self.model(torch.cat([obs, act], dim=-1))
        logits = torch.clamp(logits, -self.max_logit, self.max_logit)
        return logits

    def reward(self, obs, act):
        """Recompute reward after collected rollouts for off-policy algoritorchm"""
        with torch.no_grad():
            logits = self.forward(obs, act)
            if self.reward_type == 'GAIL':
                rewards = -torch.log(self.sigmoid(logits))
            elif self.reward_type == 'SOFTPLUS':
                rewards = -self.softplus(logits)
            elif self.reward_type == 'AIRL':
                rewards = -logits
            else:
                assert False
        return rewards


    def update(self, replay_buffer, demo_buffer):  

        agent_batch = next(replay_buffer)
        agent_obs, agent_act, _, _, _ = utils.to_torch(agent_batch, self.device)

        demo_batch = next(demo_buffer)
        demo_obs, demo_act, _, _, _ = utils.to_torch(demo_batch, self.device)

        # assert demo_states.dim() == 2
        # assert demo_states.shape[0] == self.batch_size

        obs = dict()
        for k in demo_obs.keys():
            obs[k] = torch.cat([demo_obs[k], agent_obs[k]], dim=0)
        act = torch.cat([demo_act, agent_act], dim=0)

        logits = self.forward(obs, act)
        D = self.sigmoid(logits)
        loss = self.loss(D, self.labels)

        # Backpropagation
        for encoder_opt in self.encoder_opts.values():
            encoder_opt.zero_grad(set_to_none=True)
        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()        

        grad_norms = []
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norms.append(p.grad.detach().data.norm(2))
        grad_norms = torch.stack(grad_norms)

        self.model_opt.step()
        for encoder_opt in self.encoder_opts.values():
            encoder_opt.step()

        # Log metrics
        demo_D, agent_D = D[:self.batch_size], D[self.batch_size:]
        disc_expert_acc = torch.mean((demo_D < 0.5).float())
        disc_agent_acc = torch.mean((agent_D > 0.5).float())
        disc_expert_logit = torch.mean(logits[:self.batch_size])
        disc_agent_logit = torch.mean(logits[self.batch_size:])
        metrics = {
            'disc_loss': loss.item(),
            'disc_expert_acc': disc_expert_acc.item(),
            'disc_agent_acc': disc_agent_acc.item(),  
            'disc_expert_logit': disc_expert_logit.item(),
            'disc_agent_logit': disc_agent_logit.item(),
            'grad': grad_norms.mean().item(),
        }
        return metrics