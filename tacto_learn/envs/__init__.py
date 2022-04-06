from gym.envs.registration import register

register(
    id="sawyer-gripper-v0", 
    entry_point="tacto_learn.envs.sawyer_gripper_env:make_sawyer_gripper_env",
    max_episode_steps=200,
)
