def get_service_provider(service_provider_name):
    if service_provider_name == 'victor':
        from victor.victor_services import VictorServices
        return VictorServices()
    elif service_provider_name == 'gazebo':
        from link_bot_gazebo.gazebo_services import GazeboServices
        return GazeboServices()
    elif service_provider_name == 'mujoco':
        from dm_envs.mujoco_services import MujocoServices
        return MujocoServices()
    else:
        raise NotImplementedError()
