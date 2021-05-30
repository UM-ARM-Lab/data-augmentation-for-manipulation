from arc_utilities.listener import Listener
from gazebo_msgs.msg import LinkStates


class GetLinkStates:

    def __init__(self):
        self.listener = Listener("gazebo/link_states", LinkStates)

    def get_state(self):
        return {
            'link_states': self.listener.get(),
        }