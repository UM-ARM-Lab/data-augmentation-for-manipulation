cmake_minimum_required(VERSION 3.5)
project(peter_msgs)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs trajectory_msgs message_generation)

catkin_python_setup()

add_message_files(
        FILES
        Action.msg
        AnimControllerState.msg
        AnimationControl.msg
        IntPoint2D.msg
        LabelStatus.msg
        Points.msg
        ReconstructionInputs.msg
        SubspaceDescription.msg
        Vector2.msg
        VizOptions.msg
        WheelSpeeds.msg
)

add_service_files(
        FILES
        ComputeOccupancy.srv
        DualGripperTrajectory.srv
        GetAnimControllerState.srv
        GetBool.srv
        GetDualGripperPoints.srv
        GetFloat32.srv
        GetJointState.srv
        GetOverstretching.srv
        GetPosition3D.srv
        GetRopeState.srv
        GetVizOptions.srv
        GrippersTrajectory.srv
        LinkBotReset.srv
        LinkBotState.srv
        Position3DAction.srv
        Position3DFollow.srv
        Position3DEnable.srv
        Position3DWait.srv
        Position3DStop.srv
        Position3DList.srv
        Reconstruction.srv
        RegisterPosition3DController.srv
        SetDualGripperPoints.srv
        SetRopeState.srv
        StateSpaceDescription.srv
        UnregisterPosition3DController.srv
        WorldControl.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
        trajectory_msgs)


catkin_package(CATKIN_DEPENDS geometry_msgs sensor_msgs std_msgs trajectory_msgs message_runtime)
