"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np 
from gym import spaces
from gym.core import Env


class Wrapper:
    """
    Base class for all wrappers in robosuite.

    Args:
        env (MujocoEnv): The environment to wrap.
    """

    def __init__(self, env):
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env

        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                env = env.env
            else:
                break

    def step(self, action):
        """
        By default, run the normal environment step() function

        Args:
            action (np.array): action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        return self.env.step(action)

    def reset(self):
        """
        By default, run the normal environment reset() function

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        return self.env.reset()

    def render(self, **kwargs):
        """
        By default, run the normal environment render() function

        Args:
            **kwargs (dict): Any args to pass to environment render function
        """
        return self.env.render(**kwargs)

    def observation_spec(self):
        """
        By default, grabs the normal environment observation_spec

        Returns:
            OrderedDict: Observations from the environment
        """
        return self.env.observation_spec()

    @property
    def action_spec(self):
        """
        By default, grabs the normal environment action_spec

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        return self.env.action_spec

    @property
    def action_dim(self):
        """
        By default, grabs the normal environment action_dim

        Returns:
            int: Action space dimension
        """
        return self.env.dof

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (MujocoEnv): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr


class GymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of filtered keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """
    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}

        space_dict = {}
        for k, v in self.modality_dims.items():
            if "image" in k:    
                space_dict[k] = spaces.Box(
                    low=0,
                    high=255,
                    shape=v,
                    dtype=np.uint8
                )
            else:
                space_dict[k] = spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=v, 
                    dtype=np.float32
                )
        self.observation_space = spaces.Dict(space_dict)

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _extract_obs_dict(self, obs_dict):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations

        Returns:
            obs_dict: filtered dictionary of observations
        """
        new_obs_dict = {}
        for k in self.keys:
            if k in obs_dict:
                new_obs_dict[k] = obs_dict[k]
        return new_obs_dict

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            obs_dict: filtered dictionary of observations
        """
        ob_dict = self.env.reset()
        return self._extract_obs_dict(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return filtered observation dictionary

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (dict) filtered dictionary observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._extract_obs_dict(ob_dict), reward, done, info
    
    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
