# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


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


class Actor(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim):
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        # h = self.trunk(obs)

        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim):
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # h = self.trunk(obs)
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 frame_stack):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # construct observation encoders based on observation spec
        self.encoders = {}
        self.image_key = None
        for k, shape in obs_shape.items():
            if k == 'agentview_image' or k == 'observation':
                # Multiply channel with frame_stacks
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
        self.actor = Actor(encoder_feat_dim, action_shape, hidden_dim).to(device)
        self.critic = Critic(encoder_feat_dim, action_shape, hidden_dim).to(device)
        self.critic_target = Critic(encoder_feat_dim, action_shape, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opts = {}
        for k, v in self.encoders.items():
            self.encoder_opts[k] = torch.optim.Adam(self.encoders[k].parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def set_disc_reward(self, reward_fn):
        self.disc_reward = reward_fn

    def train(self, training=True):
        self.training = training
        for encoder in self.encoders.values():
            encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

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

    def act(self, obs, step, eval_mode):
        obs = self.encode_obs(obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        for encoder_opt in self.encoder_opts.values():
            encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        for encoder_opt in self.encoder_opts.values():
            encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # image shift augment
        if self.image_key is not None:
            obs[self.image_key] = self.aug(obs[self.image_key].float())
            next_obs[self.image_key] = self.aug(next_obs[self.image_key].float())
        
        if self.disc_reward is not None:
            reward = self.disc_reward(obs, action)
            if self.use_tb:
                metrics['learned_reward'] = reward.mean().item()

        # encode
        obs = self.encode_obs(obs)
        with torch.no_grad():
            next_obs = self.encode_obs(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()


        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
