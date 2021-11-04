import numpy as np
from dm_control import composer

from link_bot.dm_envs.src.dm_envs.blocks_task import register_envs, my_blocks
from moonshine.simple_profiler import SimpleProfiler


def main():
    register_envs()

    nums_blocks = [1, 5, 10, 20]
    rtrs = []
    for num_blocks in nums_blocks:
        task = my_blocks(num_blocks)
        env = composer.Environment(task, time_limit=10, random_state=0)

        spec = env.action_spec()
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        prof = SimpleProfiler()

        def _step():
            env.step(action)

        prof.profile(200, _step)
        print(prof)
        real_time_per_step = np.percentile(prof.dts, 5)
        sim_time_per_step = env.control_timestep()
        rtr = sim_time_per_step / real_time_per_step
        print(f"real time rate: {rtr}x")

        rtrs.append(rtr)

    import matplotlib.pyplot as plt
    plt.plot(nums_blocks, rtrs)
    plt.xlabel("num blocks")
    plt.ylabel("real time rate")
    plt.show()


if __name__ == '__main__':
    main()
