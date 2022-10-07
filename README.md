# tacto_learn

## Updates
- Move environments to [robosuite fork](https://github.com/tianyudwang/robosuite)

## Installation
1. Install [this fork](https://github.com/tianyudwang/robosuite) of robosuite with custom environments that supports tactile sensing.

2.
```bash
git clone git@github.com:tianyudwang/tacto_learn.git
cd tacto_learn
pip install -e .
```

3. Training
```bash
python3 tacto_learn/scripts/train_lift.py configs/lift.yml
```

4. Monitor results
```bash
tensorboard --logdir logs
```

5. Visualize trained model
```bash
python3 tacto_learn/scripts/visualize_policy.py configs/lift.yml --trained_model ./trained_models/SAC_Lift_Panda_OSC_POSE_proprio_obj_state
```