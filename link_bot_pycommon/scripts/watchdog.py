#!/usr/bin/env python
import argparse
import signal
import subprocess
from time import perf_counter, sleep
from typing import List

import rospy
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

exit = False


def signal_handler(_, __):
    global exit
    exit = True


class Watchdog:

    def __init__(self, period: int, command: List[str], verbose: int = 0):
        global exit

        self.command = command
        self.period = period
        self.verbose = verbose
        self.n_kills = 0
        self.time_since_last_heartbeat = 0

        self.heartbeat_received = False
        self.heartbeat_subscriber = rospy.Subscriber('heartbeat', Header, self.heartbeat_callback)
        self.status_srv = rospy.Service('watchdog/status', Trigger, self.status)

        while not exit:
            print("starting:", command)
            proc = subprocess.Popen(command)

            # wait until the first message
            while not self.heartbeat_received:
                pass

            kill = False
            while not kill:
                if self.verbose > 1:
                    print("starting heartbeat timer")
                t0 = perf_counter()

                while True:
                    self.time_since_last_heartbeat = perf_counter() - t0
                    if self.verbose > 1:
                        print(f'{self.time_since_last_heartbeat:.2f}')

                    sleep(1)

                    if self.heartbeat_received:
                        if self.verbose > 1:
                            print("got heartbeat")
                        self.heartbeat_received = False
                        break
                    elif self.time_since_last_heartbeat > self.period * 1.1:  # give it some wiggle room
                        kill = True
                        break
                    elif proc.poll() is not None:
                        print("Process finished!")
                        return

            print("Missed a heartbeat! killing")
            proc.kill()
            self.n_kills += 1

    def heartbeat_callback(self, msg: Header):
        self.heartbeat_received = True

    def status(self, req: TriggerRequest):
        status_msg = f"dt={self.time_since_last_heartbeat:.1f}s, n_kills={self.n_kills}"
        return TriggerResponse(success=True, message=status_msg)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("watchdog")

    parser = argparse.ArgumentParser()
    parser.add_argument('script')
    parser.add_argument('args', nargs='*')
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    command = [args.script] + args.args
    Watchdog(args.period, command, args.verbose)


if __name__ == '__main__':
    main()
