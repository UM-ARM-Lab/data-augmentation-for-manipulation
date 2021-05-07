from time import sleep
from typing import List, Callable, Any

import halo
import numpy as np

import rospy
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import numpify
from peter_msgs.msg import AnimationControl
from peter_msgs.srv import GetAnimControllerStateRequest, GetAnimControllerState
from std_msgs.msg import Int64


class RvizAnimationController:

    def __init__(self, time_steps=None, n_time_steps: int = None):
        if time_steps is None and n_time_steps is None:
            raise ValueError("you have to pass either n_time_steps or time_steps")
        if time_steps is not None:
            self.time_steps = np.array(time_steps, dtype=np.int64)
        if n_time_steps is not None:
            self.time_steps = np.arange(n_time_steps, dtype=np.int64)
        self.command_sub = rospy.Subscriber("/rviz_anim/control", AnimationControl, self.on_control)
        self.time_pub = rospy.Publisher("/rviz_anim/time", Int64, queue_size=10)
        self.max_time_pub = rospy.Publisher("/rviz_anim/max_time", Int64, queue_size=10)
        get_srv_name = "/rviz_anim/get_state"
        self.get_state_srv = rospy.ServiceProxy(get_srv_name, GetAnimControllerState)

        rospy.logdebug(f"waiting for {get_srv_name}")
        rospy.wait_for_service(get_srv_name)
        rospy.logdebug(f"connected.")

        self.idx = 0
        self.max_idx = self.time_steps.shape[0]
        self.max_t = self.time_steps[-1]
        state_res = self.get_state_srv(GetAnimControllerStateRequest())
        self.auto_play = state_res.state.auto_play
        self.loop = state_res.state.loop
        self.period = state_res.state.period
        self.playing = self.auto_play or self.loop
        self.should_step = False
        self.fwd = True
        self.bwd = False
        self.done = False

    def on_control(self, msg: AnimationControl):
        if msg.command == AnimationControl.STEP_BACKWARD:
            self.on_bwd()
        elif msg.command == AnimationControl.STEP_FORWARD:
            self.on_fwd()
        elif msg.command == AnimationControl.PLAY_BACKWARD:
            self.on_play_backward()
        elif msg.command == AnimationControl.PLAY_FORWARD:
            self.on_play_forward()
        elif msg.command == AnimationControl.PAUSE:
            self.on_pause()
        elif msg.command == AnimationControl.DONE:
            self.on_done()
        elif msg.command == AnimationControl.SET_LOOP:
            self.loop = msg.state.loop
        elif msg.command == AnimationControl.SET_AUTO_PLAY:
            self.loop = msg.state.loop
        elif msg.command == AnimationControl.SET_PERIOD:
            self.period = msg.state.period
        elif msg.command == AnimationControl.SET_IDX:
            self.idx = msg.state.idx
            self.should_step = True
            self.fwd = False
            self.bwd = False
        else:
            raise NotImplementedError(f"Unsupported animation control {msg.command}")

    def on_fwd(self):
        self.should_step = True
        self.playing = False
        self.fwd = True
        self.bwd = False

    def on_bwd(self):
        self.should_step = True
        self.playing = False
        self.fwd = False
        self.bwd = True

    def on_play_forward(self):
        self.playing = True
        self.fwd = True
        self.bwd = False

    def on_play_backward(self):
        self.playing = True
        self.fwd = False
        self.bwd = True

    def on_pause(self):
        self.playing = False

    def on_done(self):
        self.done = True

    def step(self):
        if self.playing:
            # don't use ros time because we don't want to rely on simulation time
            sleep(self.period)
        else:
            while not self.should_step and not self.playing and not self.done:
                sleep(0.01)

        if self.fwd:
            if self.idx < self.max_idx - 1:
                self.idx += 1
            else:
                if self.auto_play:
                    self.done = True
                elif self.loop:
                    self.idx = 0
                else:
                    self.playing = False
        elif self.bwd:
            if self.idx > 0:
                self.idx -= 1
            elif self.loop:
                self.idx -= 1
                if self.idx == -1:
                    self.idx = self.max_idx - 1

        t_msg = Int64()
        t_msg.data = self.time_steps[self.idx]
        self.time_pub.publish(t_msg)

        self.should_step = False

        max_t_msg = Int64()
        max_t_msg.data = self.time_steps[-1]
        self.max_time_pub.publish(max_t_msg)

        return self.done

    def t(self):
        return self.time_steps[self.idx]


class RvizSimpleStepper:

    def __init__(self):
        self.command_sub = rospy.Subscriber("/rviz_anim/control", AnimationControl, self.on_control)
        self.should_step = False
        self.play = False

    def on_control(self, msg: AnimationControl):
        if msg.command == AnimationControl.STEP_FORWARD:
            self.should_step = True
        elif msg.command == AnimationControl.PLAY_FORWARD:
            self.should_step = True
            self.play = True
        elif msg.command == AnimationControl.PAUSE:
            self.play = False

    @halo.Halo('click step')
    def step(self):
        while not self.should_step:
            sleep(0.05)
        if not self.play:
            self.should_step = False


# pylint: disable=too-few-public-methods
class RvizAnimation:

    def __init__(self,
                 scenario: ExperimentScenario,
                 n_time_steps: int,
                 init_funcs: List[Callable],
                 t_funcs: List[Callable]):
        self.scenario = scenario
        self.init_funcs = init_funcs
        self.t_funcs = t_funcs
        self.n_time_steps = n_time_steps

    def play(self, example: Any):
        example = numpify(example)
        for init_func in self.init_funcs:
            init_func(self.scenario, example)

        controller = RvizAnimationController(n_time_steps=self.n_time_steps)
        while not controller.done:
            t = controller.t()

            for t_func in self.t_funcs:
                t_func(self.scenario, example, t)

            controller.step()
