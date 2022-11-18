import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from nn_models import Encoder, MLP, FeatureExtractor

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.obs_encoder = FeatureExtractor(obs_shape)

        self.policy = nn.Sequential(
            nn.Linear(self.obs_encoder.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Linear(hidden_dim, action_shape[0])
        self.log_std = nn.Linear(hidden_dim, action_shape[0])

        self.apply(utils.weight_init)

    def forward(self, obs):
        feat = self.obs_encoder(obs)
        feat = self.policy(feat)
        mu = self.mu(feat)
        log_std = self.log_std(feat)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = utils.TanhNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.obs_encoder = FeatureExtractor(obs_shape)

        self.Q1 = nn.Sequential(
            nn.Linear(self.obs_encoder.feat_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(self.obs_encoder.feat_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)


    def forward(self, obs, action):
        feat = self.obs_encoder(obs)
        feat_action = torch.cat([feat, action], dim=-1)
        q1 = self.Q1(feat_action)
        q2 = self.Q2(feat_action)

        return q1, q2

class SACAgent:
    def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, 
                 critic_target_tau, num_expl_steps, update_every_steps,
                 use_tb):        
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps

        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        self.target_entropy = -np.prod(action_shape)
        self.alpha = self.log_alpha.exp().item()

        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)
        self.critic = Critic(obs_shape, action_shape, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_shape, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        for k, ob in obs.items():
            obs[k] = torch.as_tensor(ob, device=self.device)
            if len(ob.shape) == 1 or len(ob.shape) == 3:
                obs[k] = obs[k].unsqueeze(0)

        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]


    def update_critic(self, obs, action, reward, discount, next_obs):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample()
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
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return metrics


    def update_actor(self, obs):
        metrics = dict()
        
        dist = self.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha * log_prob - Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics, log_prob


    def update_entropy(self, log_prob):
        metrics = dict()

        alpha = torch.exp(self.log_alpha.detach())
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        if self.use_tb:
            metrics['alpha_loss'] = alpha_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()


        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs))

        # update actor
        actor_metrics, log_prob = self.update_actor(obs)
        metrics.update(actor_metrics)

        # update entropy coefficient
        metrics.update(self.update_entropy(log_prob))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

