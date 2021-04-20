from threading import Thread
from time import sleep

import rospy
from std_msgs.msg import Header


class HeartBeat:

    def __init__(self, period: int):
        self.period = period
        self.pub = rospy.Publisher("heartbeat", Header, queue_size=10)
        self.thread = Thread(target=self.thread_main)
        self.thread.start()

    def thread_main(self):
        while True:
            sleep(self.period)
            self.pub.publish(Header(stamp=rospy.Time.now()))
