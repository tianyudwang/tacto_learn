from collections import deque, OrderedDict
from typing import Any, NamedTuple

import numpy as np

import dm_env
from dm_env import StepType, specs, TimeStep
import robosuite as suite


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class DMEWrapper(dm_env.Environment):
    def __init__(self, env, keys=None):
        self._env = env
        
        if keys is None:
            keys = []
            # Add object obs if requested
            if self._env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self._env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self._env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self._env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # robosuite returns an observation as observation specification
        ob = self._env.observation_spec()
        self._observation_spec = OrderedDict()
        for k in self.keys:
            assert k in ob.keys(), f"{k} not in robosuite observation"
            self._observation_spec[k] = specs.Array(ob[k].shape, ob[k].dtype, name=k)

        self._action_spec = specs.BoundedArray(
            (self._env.action_dim,), 
            np.float32, 
            minimum=self._env.action_spec[0],
            maximum=self._env.action_spec[1],
            name='action'
        )

    def _extract_obs(self, obs_dict):
        """
        Filter keys of interest out in observation
        """
        new_obs_dict = OrderedDict()
        for key in self.keys:
            new_obs_dict[key] = obs_dict[key]
        return new_obs_dict


    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        observation = self._env.reset()
        observation = self._extract_obs(observation)
        return TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation,
        )

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        observation = self._extract_obs(observation)
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(
            step_type=step_type, 
            reward=reward, 
            discount=1,             # discount not an attribute of robosuite MujocoEnv
            observation=observation,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='agentview_image', use_proprio=False):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key
        self._use_proprio = use_proprio

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._vis_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

        # observation_spec is a dictionary if using multimodal observation
        if self._use_proprio:
            self._proprio_key = 'robot0_proprio-state'
            assert self._proprio_key in wrapped_obs_spec
            self._obs_spec = OrderedDict()
            self._vis_spec = self._vis_spec.replace(name=self._pixels_key)
            self._obs_spec[self._pixels_key] = self._vis_spec
            self._obs_spec[self._proprio_key] = wrapped_obs_spec[self._proprio_key].replace(dtype=np.float32)
        else:
            self._obs_spec = self._vis_spec


    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        vis_obs = np.concatenate(list(self._frames), axis=0)

        if self._use_proprio:
            obs = OrderedDict()
            obs[self._pixels_key] = vis_obs
            obs[self._proprio_key] = time_step.observation[self._proprio_key].astype(np.float32)
        else:
            obs = vis_obs
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, use_proprio, seed):
    controller_configs = suite.load_controller_config(
        default_controller="OSC_POSE"
    )

    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_heights=84,
        camera_widths=84,
        controller_configs=controller_configs
    )

    env = DMEWrapper(env)

    # add wrappers
    # Robosuite env does not need action repear and action scale
    # env = ActionDTypeWrapper(env, np.float32)
    # env = ActionRepeatWrapper(env, action_repeat)
    # env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    # stack several frames
    env = FrameStackWrapper(env, frame_stack, 'agentview_image', use_proprio)
    env = ExtendedTimeStepWrapper(env)
    return env
