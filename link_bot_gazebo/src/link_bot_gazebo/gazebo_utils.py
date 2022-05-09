import contextlib
import pathlib

import halo
import psutil

import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse, GetModelStateRequest


def get_gazebo_kinect_pose(model_name="kinect2"):
    get_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    res: GetModelStateResponse = get_srv(GetModelStateRequest(model_name=model_name))
    return res.pose


def save_gazebo_pids(pids):
    with GAZEBO_PIDS_FILENAME.open('w') as f:
        for pid in pids:
            f.write(f"{pid}\n")


def get_gazebo_pids():
    look_up_new_pids = False
    if not GAZEBO_PIDS_FILENAME.exists():
        look_up_new_pids = True
    else:
        with GAZEBO_PIDS_FILENAME.open("r") as f:
            pids = [int(l.strip('\n')) for l in f.readlines()]
        if len(pids) == 0:
            look_up_new_pids = True
        else:
            for pid in pids:
                if not psutil.pid_exists(pid):
                    look_up_new_pids = True

    if look_up_new_pids:
        pids = []
        for proc in psutil.process_iter(['name', 'pid']):
            if proc.info['name'] == 'gzserver' or proc.info['name'] == 'gzclient':
                pids.append(proc.info['pid'])

    save_gazebo_pids(pids)

    return pids


GAZEBO_PIDS_FILENAME = pathlib.Path("~/.gazebo_pids").expanduser()


@halo.Halo("Getting gazebo processes")
def get_gazebo_processes():
    pids = get_gazebo_pids()
    processes = []
    for pid in pids:
        processes.append(psutil.Process(pid))
    return processes


def is_suspended():
    gazebo_processes = get_gazebo_processes()
    suspended = any([p.status == 'sleeping' for p in gazebo_processes])
    return suspended


def suspend():
    gazebo_processes = get_gazebo_processes()
    [p.suspend() for p in gazebo_processes]
    return gazebo_processes


def resume():
    gazebo_processes = get_gazebo_processes()
    [p.resume() for p in gazebo_processes]
    return gazebo_processes


@contextlib.contextmanager
def gazebo_suspended():
    initial_is_suspended = gazebo_utils.is_suspended()
    gazebo_utils.suspend()
    yield
    if not initial_is_suspended:
        gazebo_utils.resume()