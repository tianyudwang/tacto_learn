# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np 

import gym
import tacto_learn.envs

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def main():

    env = gym.make("sawyer-gripper-v0")
    print (f"Env observation space: {env.observation_space}")
    env.reset()

    # Create a hard-coded grasping policy
    policy = GraspingPolicy(env)

    # Set the initial state (obs) to None, done to False
    obs, done = None, False

    while not done:
        env.render()
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        print(obs['object']['position'], done)



if __name__ == "__main__":
    main()
