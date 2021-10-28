import numpy as np
from dm_control import composer
from dm_control import viewer

from link_bot.dm_envs.src.dm_envs.blocks_env import register_envs, my_blocks


def main():
    register_envs()

    params = {
        'num_blocks':                   10,
        'extent':                       [-0.15, 0.15, -0.15, 0.15, 1e-6, 0.15],
        'gripper_action_sample_extent': [-0.1, 0.1, -0.1, 0.1, 1e-6, 0.1],
    }
    task = my_blocks(params)
    env = composer.Environment(task, time_limit=9999, random_state=0)

    spec = env.action_spec()

    def random_policy(time_step):
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
    main()
