import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR'] = '1'

from pathlib import Path
import psutil
import json
import h5py

import hydra
import numpy as np
import torch
from dm_env import specs, StepType

import robosuite_env as rbe
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import TrainVideoRecorder, VideoRecorder

from robosuite.utils.mjcf_utils import postprocess_model_xml

torch.backends.cudnn.benchmark = True


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        print(self.cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        # set up observation/action specs
        obs_spec = self.train_env.observation_spec()
        action_spec = self.train_env.action_spec()
        assert isinstance(obs_spec, dict), "Observation must be wrapped in a dictionary"
        obs_shape = {}
        for k, v in obs_spec.items():
            obs_shape[k] = v.shape
        self.cfg.agent.obs_shape = obs_shape
        self.cfg.agent.action_shape = action_spec.shape
        self.cfg.discriminator.obs_shape = obs_shape
        self.cfg.discriminator.action_shape = action_spec.shape

        self.agent = hydra.utils.instantiate(self.cfg.agent)
        self.disc = hydra.utils.instantiate(self.cfg.discriminator)

        self.fill_expert_buffer()

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        self.train_env = rbe.make(self.cfg.env, self.cfg.frame_stack)
        self.eval_env = rbe.make(self.cfg.env, self.cfg.frame_stack)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        
        print("Observation spec:", self.train_env.observation_spec())
        self.demo_buffer = ReplayBuffer(
            self.cfg.demo_buffer_size,
            self.cfg.batch_size,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.frame_stack,
            data_specs=data_specs
        )
        self.replay_buffer = ReplayBuffer(
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.frame_stack,
            data_specs=data_specs
        )
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    def fill_expert_buffer(self):
        hdf5_path = os.path.join(hydra.utils.to_absolute_path(self.cfg.demo_dir), "demo.hdf5")
        f = h5py.File(hdf5_path, "r")   

        env_name = f["data"].attrs["env"]
        env_info = json.loads(f["data"].attrs["env_info"])

        # TODO: Check env_cfg is the same as in demo 
        demo_env = rbe.make(self.cfg.env, self.cfg.frame_stack)

        demos = list(f["data"].keys())
        for ep in demos:
            # read the model xml, using the metadata stored in the attribute for this episode
            model_xml = f["data/{}".format(ep)].attrs["model_file"]

            demo_env.reset()
            xml = postprocess_model_xml(model_xml)
            demo_env.reset_from_xml_string(xml)
            demo_env.sim.reset()

            # load the flattened mujoco states
            states = np.array(f["data/{}/states".format(ep)][()])

            # load the initial state
            demo_env.sim.set_state_from_flattened(states[0])
            demo_env.sim.forward()
            time_step = demo_env.reset(sim_reset=False)
            self.demo_buffer.add(time_step)

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            assert num_actions <= self.cfg.env.horizon, f"Demonstration length {num_actions} longer than env horizon {self.cfg.env.horizon}"

            rew = []
            for j in range(self.cfg.env.horizon):
                if j < num_actions:
                    action = actions[j]
                    time_step = demo_env.step(action)
                    self.demo_buffer.add(time_step)  
                    rew.append(time_step.reward)

                    # ensure that the actions deterministically lead to the same recorded states
                    if j < num_actions - 1:
                        state_playback = demo_env.sim.get_state().flatten()
                        if not np.allclose(states[j + 1], state_playback):
                            err = np.linalg.norm(states[j + 1] - state_playback)
                            print(f"[warning] playback diverged by {err:.6f} for ep {ep} at step {j}")  
                else:
                    # use the last time_step to pad the episode
                    # if j == self.cfg.env.horizon-1:
                    #     time_step = time_step._replace(step_type=StepType.LAST)
                    
                    # TODO: null action to pad the episode depends on the task
                    # change in position and orientation is 0, and grasp is 1
                    action = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                    time_step = demo_env.step(action)
                    self.demo_buffer.add(time_step)
                    rew.append(time_step.reward)
            print(f"Current episode return: {np.sum(rew):.2f}")
            print(f"Demo buffer contains {len(self.demo_buffer)} samples")

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_step}.mp4')

        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames)
        seed_until_step = utils.Until(self.cfg.num_seed_frames)
        eval_every_step = utils.Every(self.cfg.eval_every_frames)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_step}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step
                    with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

                print(f"Available RAM: {psutil.virtual_memory().available / 1e9} GB")

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_step)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent and discriminator
            if not seed_until_step(self.global_step):

                # Train policy with reward from discriminator
                self.agent.set_disc_reward(self.disc.reward)
                metrics = self.agent.update(self.replay_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_step, ty='train_agent')

                metrics = self.disc.update(self.replay_buffer, self.demo_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_step, ty='train_disc')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()