import numpy as np

from tacto_learn.models.policy import GraspingPolicy
from tacto_learn.utils import types

def collect_trajectory(env, policy):    
    obs, acts, rews, dones = [], [], [], []
    ob, done = env.reset(), False
    if isinstance(type(policy), GraspingPolicy):
        print("Resetting grasping policy")
        policy.reset()
    while not done:
        act = policy.predict(ob)
        obs.append(ob)
        acts.append(act)

        ob, rew, done, info = env.step(act)
        rews.append(rew)
        dones.append(done)

    obs = types.convert_observation_dict(obs)
    trajectory = types.Trajectory(
        observations=obs, 
        actions=np.array(acts), 
        rewards=np.array(rews)
    )
    return trajectory

def collect_trajectories(env, policy, batch_size):
    trajectories = []
    for _ in range(batch_size):
        trajectories.append(collect_trajectory(env, policy))
    return trajectories

def log_metrics(logger, metrics, namespace):
    for k, v in metrics.items():
        if v.dim() < 1 or (v.dim() == 1 and v.shape[0] <= 1):
            logger.record_mean(f"{namespace}/{k}", v.item())
        else:
            logger.record_mean(f"{namespace}/{k}Max", th.amax(v).item())
            logger.record_mean(f"{namespace}/{k}Min", th.amin(v).item())
            logger.record_mean(f"{namespace}/{k}Mean", th.mean(v).item())
            logger.record_mean(f"{namespace}/{k}Std", th.std(v).item())