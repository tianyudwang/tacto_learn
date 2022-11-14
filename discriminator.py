import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.utils.parametrizations as p

import utils
from nn_models import Encoder, MLP, FeatureExtractor


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
        reward_type,
        update_every_steps,
        spectral_norm
        # clip_reward_range: Optional[float] = -1.0,
    ):
        # super().__init__()

        self.device = device
        self.reward_type = reward_type
        self.max_logit = max_logit
        self.batch_size = batch_size
        self.update_every_steps = update_every_steps
        # self.max_grad_norm = max_grad_norm


        # discriminator
        self.obs_encoder = FeatureExtractor(obs_shape, spectral_norm=spectral_norm).to(device)
        self.disc = MLP(
            self.obs_encoder.feat_dim+action_shape[0], 
            1, 
            spectral_norm=spectral_norm
        ).to(device)

        # optimizers
        self.disc_opt = torch.optim.Adam(
            list(self.obs_encoder.parameters())+list(self.disc.parameters()), 
            lr=lr
        )

        self.sigmoid = nn.Sigmoid()
        # self.softplus = nn.Softplus()
        self.loss = nn.BCELoss()
        
        # D = 0 for expert and D = 1 for agent
        demo_labels = torch.zeros((batch_size, 1), device=self.device)
        agent_labels = torch.ones((batch_size, 1), device=self.device) 
        self.labels = torch.cat((demo_labels, agent_labels), dim=0)

    def forward(self, obs, act):
        feat = self.obs_encoder(obs)
        logits = self.disc(torch.cat([feat, act], dim=-1))
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


    def update(self, replay_buffer, demo_buffer, step):  

        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        agent_batch = next(replay_buffer)
        agent_obs, agent_act, _, _, _ = utils.to_torch(agent_batch, self.device)

        demo_batch = next(demo_buffer)
        demo_obs, demo_act, _, _, _ = utils.to_torch(demo_batch, self.device)

        obs = dict()
        for k in demo_obs.keys():
            obs[k] = torch.cat([demo_obs[k], agent_obs[k]], dim=0)
        act = torch.cat([demo_act, agent_act], dim=0)

        logits = self.forward(obs, act)
        D = self.sigmoid(logits)
        loss = self.loss(D, self.labels)

        # Backpropagation
        self.disc_opt.zero_grad(set_to_none=True)
        loss.backward()        

        grad_norms = []
        for p in list(filter(lambda p: p.grad is not None, list(self.disc.parameters()) + list(self.obs_encoder.parameters()) )):
            grad_norms.append(p.grad.detach().data.norm(2))
        grad_norms = torch.stack(grad_norms)

        self.disc_opt.step()

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