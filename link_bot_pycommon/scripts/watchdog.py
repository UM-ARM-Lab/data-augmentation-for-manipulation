import argparse
import signal
import subprocess
from multiprocessing import Queue
from time import sleep

import rospy
from std_msgs.msg import Header

exit = False
heartbeat_queue = Queue()


def signal_handler(_, __):
    global exit
    exit = True


def heartbeat_callback(msg: Header):
    print("got heartbeat")
    heartbeat_queue.put(msg)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("watchdog")

    parser = argparse.ArgumentParser()
    parser.add_argument('script')
    parser.add_argument('args', nargs='*')
    parser.add_argument('--period', type=int, default=2)

    args = parser.parse_args()

    heartbeat_subscriber = rospy.Subscriber('heartbeat', Header, heartbeat_callback)

    command = [args.script] + args.args

    while not exit:
        print("starting:", command)
        proc = subprocess.Popen(command)

        sleep(2 * args.period)

        while True:
            sleep(args.period)
            if heartbeat_queue.empty():
                break
            else:
                heartbeat_queue.get()

        print("Missed a heartbeat! killing")
        proc.kill()


if __name__ == '__main__':
    main()
