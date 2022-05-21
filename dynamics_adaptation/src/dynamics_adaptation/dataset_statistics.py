import numpy as np

from link_bot_pycommon.func_list_registrar import FuncListRegistrar

metrics_funcs = FuncListRegistrar()


@metrics_funcs
def gripper_distance(example):
    assert example['left_gripper'].ndim == 1
    return np.linalg.norm(example['left_gripper'] - example['right_gripper'], axis=-1)


@metrics_funcs
def gripper_distance_z(example):
    assert example['left_gripper'].ndim == 1
    return (example['left_gripper'] - example['right_gripper'])[2]


@metrics_funcs
def avg_x(example):
    assert example['left_gripper'].ndim == 1
    return np.mean(np.concatenate([
        example['left_gripper'][0:1],
        example['right_gripper'][0:1],
        example['rope'][0::3],
    ], -1))


@metrics_funcs
def avg_y(example):
    assert example['left_gripper'].ndim == 1
    return np.mean(np.concatenate([
        example['left_gripper'][1:2],
        example['right_gripper'][1:2],
        example['rope'][1::3],
    ], -1))


@metrics_funcs
def avg_z(example):
    assert example['left_gripper'].ndim == 1
    return np.mean(np.concatenate([
        example['left_gripper'][2:3],
        example['right_gripper'][2:3],
        example['rope'][2::3],
    ], -1))
