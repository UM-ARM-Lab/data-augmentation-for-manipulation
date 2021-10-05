from time import sleep
from typing import List, Callable, Any

import halo
import numpy as np

import rospy
from peter_msgs.msg import AnimationControl
from peter_msgs.srv import GetAnimControllerStateRequest, GetAnimControllerState
from rosgraph.names import ns_join
from std_msgs.msg import Int64


class RvizAnimationController:

    def __init__(self, time_steps=None, n_time_steps: int = None, ns: str = 'rviz_anim'):
        self.ns = ns
        if time_steps is None and n_time_steps is None:
            raise ValueError("you have to pass either n_time_steps or time_steps")
        if time_steps is not None:
            self.time_steps = np.array(time_steps, dtype=np.int64)
        if n_time_steps is not None:
            self.time_steps = np.arange(n_time_steps, dtype=np.int64)
        self.command_sub = rospy.Subscriber(ns_join(ns, "control"), AnimationControl, self.on_control)
        self.time_pub = rospy.Publisher(ns_join(ns, "time"), Int64, queue_size=10)
        self.max_time_pub = rospy.Publisher(ns_join(ns, "max_time"), Int64, queue_size=10)
        get_srv_name = ns_join(ns, "get_state")
        self.get_state_srv = rospy.ServiceProxy(get_srv_name, GetAnimControllerState)

        rospy.logdebug(f"waiting for {get_srv_name}")
        rospy.wait_for_service(get_srv_name)
        rospy.logdebug(f"connected.")

        self.max_idx = self.time_steps.shape[0]
        self.max_t = self.time_steps[-1]
        self.fwd = True
        self.bwd = False
        self.should_step = False

        state_res = self.get_state_srv(GetAnimControllerStateRequest())
        self.auto_play = state_res.state.auto_play
        self.done_after_playing = state_res.state.done_after_playing
        self.loop = state_res.state.loop
        self.period = state_res.state.period

        self.reset()

        self.update_state_periodically = rospy.Timer(rospy.Duration(nsecs=100_000_000), self.update_state)

    def __repr__(self):
        return self.ns

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.idx = 0
        self._done = False
        self.playing = self.auto_play or self.loop

    def update_state(self, _):
        try:
            state_res = self.get_state_srv(GetAnimControllerStateRequest())
            self.auto_play = state_res.state.auto_play
            self.loop = state_res.state.loop
            self.period = state_res.state.period
        except rospy.ServiceException:
            pass

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
        elif msg.command == AnimationControl.SET_DONE_AFTER_PLAYING:
            self.done_after_playing = msg.state.done_after_playing
        elif msg.command == AnimationControl.SET_AUTO_PLAY:
            self.auto_play = msg.state.auto_play
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

    @property
    def done(self):
        return self._done

    def on_done(self):
        self._done = True

    def step(self):
        self.wait()
        self.update_idx()
        self.publish_updated_idx()
        return self._done

    def publish_updated_idx(self):
        t_msg = Int64()
        t_msg.data = self.time_steps[self.idx]
        self.time_pub.publish(t_msg)
        self.should_step = False
        max_t_msg = Int64()
        max_t_msg.data = self.time_steps[-1]
        self.max_time_pub.publish(max_t_msg)

    def update_idx(self):
        if self.fwd:
            if self.idx < self.max_idx - 1:
                self.idx += 1
            else:
                if self.done_after_playing:
                    self._done = True
                    self.playing = False
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
            elif self.idx == 0:
                self.playing = False

    def wait(self):
        if self.playing:
            # don't use ros time because we don't want to rely on simulation time
            sleep(self.period)
        else:
            while not self.should_step and not self.playing and not self._done:
                sleep(0.01)

    def t(self):
        return self.time_steps[self.idx]


class RvizSimpleStepper:

    def __init__(self, ns='rviz_anim'):
        self.command_sub = rospy.Subscriber(ns_join(ns, "control"), AnimationControl, self.on_control)
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
                 myobj,
                 n_time_steps: int,
                 init_funcs: List[Callable],
                 t_funcs: List[Callable],
                 ns='rviz_anim'):
        self.myobj = myobj
        self.init_funcs = init_funcs
        self.t_funcs = t_funcs
        self.n_time_steps = n_time_steps
        self.ns = ns

    def play(self, example: Any):
        print("Warning: you may need to call numpify!")
        for init_func in self.init_funcs:
            init_func(self.myobj, example)

        controller = RvizAnimationController(n_time_steps=self.n_time_steps, ns=self.ns)
        while not controller.done:
            t = controller.t()

            for t_func in self.t_funcs:
                t_func(self.myobj, example, t)

            controller.step()


class MultiRvizAnimationController:

    def __init__(self, sub_anims: List[RvizAnimationController]):
        self.sub_anims = sub_anims
        self.sub_anims_rev = sub_anims[::-1]

    def t(self):
        return [c.t() for c in self.sub_anims]

    @property
    def done(self):
        return self.sub_anims[0].done

    def step(self):
        while True:
            for j, c in enumerate(self.sub_anims_rev):
                # if c is the highest level anim, we just use False
                parent_at_last_idx = False
                if j + 1 < len(self.sub_anims_rev):
                    parent = self.sub_anims_rev[j + 1]
                    parent_at_last_idx = parent.idx == parent.max_t
                if c.done and not parent_at_last_idx:
                    c.reset()
                elif c.playing or c.should_step:
                    if c.playing:
                        sleep(c.period)
                    c.update_idx()
                    c.publish_updated_idx()
                    return

            sleep(0.1)  # prevent eating CPU
