from typing import Dict, List

import numpy as np

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from merrrt_visualization.rviz_animation_controller import RvizAnimationController


class AnimatableScenario(ExperimentScenario):
    def animate_final_path(self,
                           environment: Dict,
                           planned_path: List[Dict],
                           actions: List[Dict]):
        time_steps = np.arange(len(planned_path))
        self.plot_environment_rviz(environment)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            s_t_planned = planned_path[t]
            self.plot_state_rviz(s_t_planned, label='planned', color='#FF4616')
            if len(actions) > 0:
                if t < anim.max_t:
                    self.plot_action_rviz(s_t_planned, actions[t])
                else:
                    self.plot_action_rviz(planned_path[t - 1], actions[t - 1])

            anim.step()
