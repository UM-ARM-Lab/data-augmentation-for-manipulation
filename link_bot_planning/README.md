# To run the simulated rope experiment

```
ROS_NAMESPACE=hdt_michigan roslaunch hdt_michigan_moveit planning_context.launch
rosrun link_bot_gazebo relaunch_gazebo.py val.launch car.world
./scripts/iterative_fine_tuning.py /media/shared/ift/small-hooks-diverse-aug-9
# then respond to the prompts as follows:
ift_config_filename: ift_configs/full_method.hjson
seed: 0  # you can change this, our method is robust to random seed :)
initial_classifier_checkpoint [/media/shared/cl_trials/untrained-1/August_13_17-03-09_45c09348d1]:
initial_recovery_checkpoint [None]:
planner_params_filename [planner_configs/val_car/random_accept.hjson]:
test_scenes_dir [test_scenes/car0-sm]: test_scenes/small-hooks-diverse
test_scenes_indices [0]: null
```
