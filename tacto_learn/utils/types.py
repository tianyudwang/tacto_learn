from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import torch as th


def convert_observation_dict(obs: List[Dict[str, np.ndarray]]):
    new_obs = {}
    keys = obs[0].keys()
    for key in keys:
        new_obs[key] = np.stack([ob[key] for ob in obs], axis=0)
        if 'depth' in key or 'gripper_width' in key:
            new_obs[key] = new_obs[key][..., np.newaxis]
    return new_obs


@dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""
    
    observations: Dict[str, np.ndarray]
    """Observations, each observation type has shape (trajectory_len, ) + observation_shape."""

    actions: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    rewards: np.ndarray
    """Rewards, shape (trajectory_len, )"""
    
    def __len__(self):
        """Returns number of transitions, equal to the number of actions."""
        return len(self.actions)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.actions) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")

        for k, v in self.observations.items():
            if not len(v) == len(self.actions):
                raise ValueError(f"Observations {k} length {len(self.observations)}, actions length {len(self.actions)}")

            if 'color' in k or 'depth' in k:
                if not len(v.shape) == 4:
                    raise ValueError(f"Image observation {k} has shape {v.shape}, maybe missing channel dimension")
            else:
                if not len(v.shape) == 2:
                    raise ValueError(f"Vector observation {k} has shape {v.shape}")

@dataclass(frozen=True)
class Transition:
    observation: Dict[str, np.ndarray]
    """Observation, each observation type has shape (observation_shape, )."""

    action: np.ndarray
    """Action, shape (action_shape, )."""

    reward: np.ndarray
    """Reward, shape (1, )."""

    def __len__(self):
        """Length of a transition is always 1"""
        return 1

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        for k, v in self.observation.items():
            if 'color' in k or 'depth' in k:
                if not len(v.shape) == 3:
                    raise ValueError(f"Image observation {k} has shape {v.shape}, maybe missing channel dimension")
            else:
                if not len(v.shape) == 1:
                    raise ValueError(f"Vector observation {k} has shape {v.shape}")

def convert_trajectories_to_transitions(trajectories: List[Trajectory]) -> List[Transition]:
    """Flatten a series of trajectories to a series of transitions"""
    assert len(trajectories) >= 1, "Cannot convert empty trajectory"

    transitions = []
    for traj in trajectories:
        for i in range(len(traj)):

            ob = {}
            for k in traj.observations.keys():
                ob[k] = traj.observations[k][i]

            transition = Transition(
                observation=ob, 
                action=traj.actions[i], 
                reward=traj.rewards[i]
            )
            transitions.append(transition)

    assert len(transitions) == sum([len(traj) for traj in trajectories]), (
        "Number of transitions does not match after conversion"
    )
    return transitions