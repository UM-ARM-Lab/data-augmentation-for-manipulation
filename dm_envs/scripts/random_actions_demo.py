from dm_control import composer
from dm_control import viewer

from dm_envs.cylinders_task import PlanarPushingCylindersTask


def main():
    params = {
        'num_blocks':                   1,
        'num_objs':                     1,
        'block_size':                   0.03,
        'height':                       0.04,
        'radius':                       0.02,
        'extent':                       [-0.15, 0.15, -0.15, 0.15, 1e-6, 0.15],
        'gripper_action_sample_extent': [-0.1, 0.1, -0.1, 0.1, 1e-6, 0.1],
        'blocks_init_extent':           [-0.1, 0.1, -0.1, 0.1, 1e-6, 0.1],
        'objs_init_extent':             [-0.1, 0.1, -0.1, 0.1, 1e-6, 0.05],
    }
    # task = PlanarPushingBlocksTask(params)
    task = PlanarPushingCylindersTask(params)
    env = composer.Environment(task, time_limit=9999, random_state=0)

    spec = env.action_spec()

    # # restore state!
    # import pickle
    # with open("data.pkl", 'rb') as f:
    #     data = pickle.load(f)
    # mjdata = data['state']
    # env._physics._reload_from_data(mjdata)

    def random_policy(time_step):
        # return data['actions'][0]
        return np.zeros(6)

    viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
    main()
