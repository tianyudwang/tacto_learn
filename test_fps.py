import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper


def make_env():
    # create environment instance
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    env = GymWrapper(env)
    return env

def test_one_env():
    env = make_env()
    for i in range(3):
        # reset the environment
        start_time = time.time()
        env.reset()     

        for i in range(1000):
            action = np.random.randn(env.robots[0].dof) # sample random action
            obs, reward, done, info = env.step(action)  # take action in the environment
        
        duration = time.time() - start_time
        print(f"fps: {1000 / duration:.2f}")

if __name__ == '__main__':
    test_one_env()