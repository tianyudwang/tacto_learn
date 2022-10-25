import psutil
import numpy as np
from replay_buffer import AbstractReplayBuffer


class EfficientReplayBuffer(AbstractReplayBuffer):
    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack,
                 data_specs=None):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = 0
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        # fixed since we can only sample transitions that occur nstep earlier
        # than the end of each episode or the last recorded observation
        self.discount_vec = np.power(discount, np.arange(nstep)).astype('float32')
        self.next_dis = discount**nstep

        mem_available = psutil.virtual_memory().available

        obs_spec, act_spec = data_specs[0], data_specs[1]
        assert isinstance(obs_spec, dict), "Observation must be wrapped in a dictionary"
        self.obs_shape = {k: v.shape for k, v in obs_spec.items()}
        self.obs = {}
        for k, shape in self.obs_shape.items():
            # images, assumes channel-first
            if len(shape) == 3:
                self.ims_channels = shape[0] // self.frame_stack
                self.obs[k] = np.zeros(
                    [self.buffer_size, self.ims_channels, *shape[1:]], 
                    dtype=np.uint8
                )
            # proprio state
            elif len(shape) == 1:
                self.obs[k] = np.zeros([self.buffer_size, *shape], dtype=np.float32)
            else:
                raise ValueError(f"Invalid observation shape {shape} for {k}")
        # else:
        #     self.obs_shape = obs_spec.shape
        #     self.ims_channels = self.obs_shape[0] // self.frame_stack
        #     self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)

        self.act_shape = act_spec.shape
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        # which timesteps can be validly sampled (Not within nstep from end of
        # an episode or last recorded observation)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

        # calculate memory usage
        total_memory_usage = 0
        if isinstance(self.obs, dict):
            total_memory_usage += sum([v.nbytes for v in self.obs.values()])
        else:
            total_memory_usage += self.obs.nbytes
        total_memory_usage += self.act.nbytes + self.rew.nbytes + self.dis.nbytes + self.valid.nbytes


        if total_memory_usage > mem_available:
            assert False, f"Total memory usage {total_memory_usage/1e9:.2f} GB larger than available memory {mem_available/1e9:.2f} GB"
        else:
            print(f"Replay buffer using {total_memory_usage/1e9:.2f} GB out of {mem_available/1e9:.2f} GB available memory")

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def _extract_obs(self, obs):
        """
        Extracts latest timestep observation 
        where images may include frame_stack of previous timesteps
        """
        assert isinstance(obs, dict), "Observation must be wrapped in a dictionary"
        latest_obs = {}
        for k, v in obs.items():
            if self._is_image(v):
                latest_obs[k] = v[-self.ims_channels:]
            else:
                latest_obs[k] = v
        return latest_obs

    def _is_image(self, obs):
        """Check if observation is an image"""
        return len(obs.shape) >= 3

    def _copy_obs_to_buffer(self, obs, index):
        """
        Copy observation into buffer at given index location
        """
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.obs[k][index] = v
        else:
            self.obs[index] = obs

    def add(self, time_step):
        first = time_step.first()
        latest_obs = self._extract_obs(time_step.observation)
        if first:
            # if first observation in a trajectory, record frame_stack copies of it
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    # self.obs[self.index:self.buffer_size] = latest_obs
                    # self.obs[0:end_index] = latest_obs
                    for index in range(self.index, self.buffer_size):
                        self._copy_obs_to_buffer(latest_obs, index)
                    for index in range(end_index):
                        self._copy_obs_to_buffer(latest_obs, index)
                    self.full = True
                else:
                    # self.obs[self.index:end_index] = latest_obs
                    for index in range(self.index, end_index):
                        self._copy_obs_to_buffer(latest_obs, index)
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                # self.obs[self.index:end_index] = latest_obs
                for index in range(self.index, end_index):
                    self._copy_obs_to_buffer(latest_obs, index)
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            self._copy_obs_to_buffer(latest_obs, self.index)
            self.act[self.index] = time_step.action
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def __next__(self, ):
        # sample only valid indices
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack(
            [np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
            for i in range(n_samples)], axis=0
        ) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement reward computation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        if isinstance(self.obs_shape, dict):
            obs, nobs = {}, {}
            for k, shape in self.obs_shape.items():
                # gather images with frame stacks
                if len(shape) == 3:
                    obs[k] = np.reshape(self.obs[k][obs_gather_ranges], [n_samples, *shape])
                    nobs[k] = np.reshape(self.obs[k][nobs_gather_ranges], [n_samples, *shape])
                # extract proprio states
                else:
                    obs[k] = self.obs[k][obs_gather_ranges[:,-1]]
                    nobs[k] = self.obs[k][nobs_gather_ranges[:,-1]]
        else:
            obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
            nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs, act, rew, dis, nobs)
        return ret

