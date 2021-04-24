from threading import Thread
from time import sleep

import rospy
from std_msgs.msg import Header


class HeartBeat:

    def __init__(self, period: int = 10):
        self.period = period
        self.pub = rospy.Publisher("heartbeat", Header, queue_size=10)
        self.thread = Thread(target=self.thread_main)
        self.counter = 0

    def start(self):
        self.thread.start()

    def thread_main(self):
        while not rospy.is_shutdown():
            sleep(self.period)
            self.pub.publish(Header(stamp=rospy.Time.now()))

    def update(self):
        if self.counter == 10:
            self.counter = 0
            self.pub.publish(Header(stamp=rospy.Time.now()))
