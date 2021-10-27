import numpy as np
from dm_control import manipulation

from dm_control import viewer

from link_bot.dm_envs.src.dm_envs.blocks_env import register_envs


def main():
    register_envs()

    env = manipulation.load('my_blocks', seed=0)

    spec = env.action_spec()

    def random_policy(time_step):
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
    main()
