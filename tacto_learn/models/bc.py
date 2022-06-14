from tqdm import tqdm

import numpy as np
import torch as th
from torch import optim
from torch import nn
from stable_baselines3.common.logger import configure

from tacto_learn.models.policy import MultiModalPolicy
from tacto_learn.utils.replay_buffer import ReplayBuffer
from tacto_learn.utils import utils
from tacto_learn.utils import pytorch_utils as ptu
from .feature_extractors import CustomCombinedExtractor

def lr_schedule(t):
    return 3e-4

class BC:
    def __init__(self, env, expert_policy):

        ptu.init_gpu()

        self.env = env
        self.expert_policy = expert_policy
        self.replay_buffer = ReplayBuffer()

        # self.batch_size = 1
        # self.logger = configure('/tmp/tacto_BC/', ["tensorboard"])
        self.logger = configure('/home/siddhant/tacto_ws/tensorboard_logs/BC/full_obs/test', ["tensorboard"])
        
        # policy_cfg = None
        self.policy = MultiModalPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=CustomCombinedExtractor,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=3e-4,
        )
        self.ent_weight = 1e-3

    def train(self):
        demonstrations = utils.collect_trajectories(self.env, self.expert_policy, 3)
        returns = [np.sum(traj.rewards) for traj in demonstrations]
        print(f"Demonstration return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
        self.replay_buffer.add_rollouts(demonstrations)

        for i in tqdm(range(10000)):
            if i % 1000 == 0:
                self.evaluate()
            
            self.update()
            self.logger.dump(i)

    def evaluate(self):
        trajectories = utils.collect_trajectories(self.env, self.policy, 2)
        returns = [np.sum(traj.rewards) for traj in trajectories]
        # lengths = [len(traj) for traj in trajectories]

        print(f"Eval return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

    def update(self):
        # trajectories_batch = self.replay_buffer.sample_random_trajectories(4)
        transitions = self.replay_buffer.sample_random_transitions(512)
        obs, acts = self.preprocess(transitions)
        
        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        # l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss #+ l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            'loss': loss,
            'log_prob': log_prob,
            'entropy': entropy
        }
        utils.log_metrics(self.logger, metrics, 'BC')


    # def preprocess_batch(self, trajectories_batch):
    #     obs = {}
    #     keys = trajectories_batch[0].observations.keys()
    #     for k in keys:
    #         v = np.concatenate([traj.observations[k] for traj in trajectories_batch], axis=0)
    #         obs[k] = ptu.from_numpy(v)
    #         if 'color' in k or 'depth' in k:
    #             obs[k] = obs[k].permute(0, 3, 1, 2)
    #     acts = np.concatenate([traj.actions for traj in trajectories_batch], axis=0)
    #     acts = ptu.from_numpy(acts)
    #     return obs, acts

    def preprocess(self, transitions):
        obs = {}
        keys = transitions[0].observation.keys()
        for k in keys:
            v = np.stack([tran.observation[k] for tran in transitions], axis=0)
            obs[k] = ptu.from_numpy(v)
            if 'color' in k or 'depth' in k:
                obs[k] = obs[k].permute(0, 3, 1, 2)
        acts = np.stack([tran.action for tran in transitions], axis=0)
        acts = ptu.from_numpy(acts)
        return obs, acts