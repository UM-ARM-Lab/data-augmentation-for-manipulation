#!/usr/bin/env python
import argparse
import signal
import subprocess
from time import perf_counter, sleep

import rospy
from std_msgs.msg import Header

exit = False
heartbeat_received = False


def signal_handler(_, __):
    global exit
    exit = True


def heartbeat_callback(msg: Header):
    global heartbeat_received
    heartbeat_received = True


def main():
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("watchdog")

    parser = argparse.ArgumentParser()
    parser.add_argument('script')
    parser.add_argument('args', nargs='*')
    parser.add_argument('--period', type=int, default=1)

    args = parser.parse_args()

    heartbeat_subscriber = rospy.Subscriber('heartbeat', Header, heartbeat_callback)
    global heartbeat_received

    command = [args.script] + args.args

    while not exit:
        print("starting:", command)
        proc = subprocess.Popen(command)

        # wait until the first message
        while not heartbeat_received:
            pass

        kill = False
        while not kill:
            print("starting heartbeat timer")
            t0 = perf_counter()

            while True:
                time_since_last_heartbeat = perf_counter() - t0
                print(f'{time_since_last_heartbeat:.2f}')

                sleep(1)

                if heartbeat_received:
                    print("got heartbeat")
                    heartbeat_received = False
                    break
                elif time_since_last_heartbeat > args.period * 1.1:  # give it some wiggle room
                    kill = True
                    break

        print("Missed a heartbeat! killing")
        proc.kill()


if __name__ == '__main__':
    main()
