import psutil
import numpy as np
from replay_buffer import AbstractReplayBuffer


class NStepReplayBuffer(AbstractReplayBuffer):
    """
    Replay buffer with options for n-step sampling and frame-stacking images
    Environment should not be frame-stacked to save memory
    Instead frame stack is performed when sampling
    Currently it does not support wrapped-around sampling, 
    e.g. obs at self.buffer_size and nobs at self.nstep-1
    """
    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack,
                 data_specs=None):
        self.buffer_size = buffer_size
        self.index = 0                  # index in replay buffer
        self.traj_index = 0             # index in current trajectory
        self.frame_stack = frame_stack
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
        self.obs_shapes = {k: v.shape for k, v in obs_spec.items()}
        self.obs = {}
        for k, shape in self.obs_shapes.items():
            # images, assumes channel-first
            if len(shape) == 3:
                assert shape[0] <= 12, "Image should be channel-first"
                c, h, w = shape
                self.im_channels = c // frame_stack
                self.obs[k] = np.zeros([self.buffer_size, self.im_channels, h, w], dtype=np.uint8)
            # proprio/object state
            elif len(shape) == 1:
                self.obs[k] = np.zeros([self.buffer_size, *shape], dtype=np.float32)
            else:
                raise ValueError(f"Invalid observation shape {shape} for {k}")

        self.act = np.zeros([self.buffer_size, *act_spec.shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)

        # which timesteps can be validly sampled
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)
        # decide which frames to stack
        self.frame_indices = np.zeros([self.buffer_size, self.frame_stack], dtype=np.uint32)

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

    def _copy_obs_to_buffer(self, obs):
        """
        Copy observation into buffer at current index location
        Only keep the image observation at current time step
        """
        assert isinstance(obs, dict), "Observation must be wrapped in a dictionary"
        for k, shape in self.obs_shapes.items():
            if len(shape) == 3:
                self.obs[k][self.index] = obs[k][-self.im_channels:].copy()
            else:
                self.obs[k][self.index] = obs[k].copy()

    def add(self, time_step):

        # Skip the last timestep
        # Not really necessary to skip the last time step but 
        if time_step.last():
            # Check buffer size is divisible by episode length so that 
            # there is no wrapped around episode stored in buffer
            assert self.buffer_size % self.traj_index == 0, (
                f"Buffer size {self.buffer_size} should be a integer multiple of episode length {self.traj_index+1}"
            )
            return

        # Skip the first timestep
        if time_step.first():
            self.traj_index = 0

        self._copy_obs_to_buffer(time_step.observation)
        self.act[self.index] = time_step.action.copy()
        self.rew[self.index] = time_step.reward
        self.dis[self.index] = time_step.discount


        # cannot sample at t + self.nstep when buffer at t is filled
        self.valid[(self.index + self.nstep) % self.buffer_size] = False

        # current index can be sampled if n_steps before it is filled
        if self.traj_index >= self.nstep:
            self.valid[self.index] = True

        # determine frame indices for stacking
        if self.traj_index + 1 >= self.frame_stack:
            indices = np.arange(self.index-self.frame_stack, self.index, dtype=np.uint32)+1
        else:
            # pad first image of trajectory
            indices = np.concatenate([
                np.ones(self.frame_stack-self.traj_index-1, dtype=np.uint32) * (self.index-self.traj_index),
                np.arange(self.index-self.traj_index, self.index+1, dtype=np.uint32)
            ])
            assert indices.shape[0] == self.frame_stack
        self.frame_indices[self.index] = indices

        self.index += 1
        self.traj_index += 1
        if self.index == self.buffer_size:
            self.full = True 
            self.index = 0

    def __next__(self):
        # sample only valid indices
        # Does not sample wrapped around indices, 
        # e.g. obs at self.buffer_size and nobs at 
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)


    def gather_nstep_indices(self, indices):
        """
        Should sample s_t, a_t, r_{t:t+n-1}, s_{t+n}
        The indices variable points to the next observation location
        Note that observation indices are shifted by 1 in buffer
        observations at t-self.nstep with self.frame_stack
        next observations at t with self.frame_stack
        actions at t-self.nstep+1
        rewards and discounts from t-nstep+1 to t
        """

        n_samples = indices.shape[0]

        ranges = np.stack(
            [np.arange(indices[i]-self.nstep, indices[i]) + 1 for i in range(n_samples)],
            axis=0
        )

        act = self.act[indices-self.nstep+1]
        rews = self.rew[ranges]
        rew = np.sum(rews * self.discount_vec, axis=1, keepdims=True)

        obs, nobs = {}, {}
        for k, shape in self.obs_shapes.items():
            # gather images with frame stacks
            if len(shape) == 3:
                obs_frame_indices = self.frame_indices[indices-self.nstep]
                nobs_frame_indices = self.frame_indices[indices]
                obs[k] = self.obs[k][obs_frame_indices].reshape(n_samples, -1, shape[1], shape[2])
                nobs[k] = self.obs[k][nobs_frame_indices].reshape(n_samples, -1, shape[1], shape[2])
            else:
                obs[k] = self.obs[k][indices-self.nstep]
                nobs[k] = self.obs[k][indices]

        # TODO: implement discount
        dis = self.next_dis

        return obs, act, rew, dis, nobs