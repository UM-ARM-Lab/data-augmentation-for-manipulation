#!/usr/bin/env python
import argparse
import pickle

import numpy as np
import pathlib

import moveit.core
from arc_utilities import ros_init
from moveit.core import collision_detection
import rospkg
from moveit.core.planning_scene import PlanningScene

def generate_grid_points(extent):
    return np.meshgrid(x, y, z)

@ros_init.with_ros("generate_robot_pointcloud")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfilename', type=pathlib.Path)

    # TODO: use job chunker?

    rospack = rospkg.RosPack()
    description_pkg = rospack.get_path('hdt_michigan_description')
    moveit_config_pkg = rospack.get_path('hdt_michigan_moveit')

    urdf_path = pathlib.Path(description_pkg) / 'urdf' / 'hdt_michigan.urdf'
    srdf_path = pathlib.Path(moveit_config_pkg) / 'config' / 'hdt_michigan.srdf'

    kinematic_model = moveit.core.load_robot_model(urdf_path.as_posix(), srdf_path.as_posix())
    planning_scene = PlanningScene(kinematic_model, collision_detection.World())

    collision_request = collision_detection.CollisionRequest()
    collision_result = collision_detection.CollisionResult()
    collision_request.contacts = True
    collision_request.max_contacts = 1000
    acm = planning_scene.getAllowedCollisionMatrix()
    state = planning_scene.getCurrentState()

    # TODO: this script would need API to add an object to the planning scene, which isn't implemented yet
    collision_sphere_link_name = 'collision_sphere'

    links = kinematic_model.getLinkModelNames()
    points = {}
    for link_to_check in links:
        # setup ACM to only check collision with a particular link
        for other_link in links:
            acm.setEntry(other_link, collision_sphere_link_name, False)
        acm.setEntry(link_to_check, collision_sphere_link_name, True)

        points[link_to_check] = []
        for x, y, z in generate_grid_points():
            collision_result.clear()

            planning_scene.checkCollision(collision_request, collision_result, state, acm)
            assert len(collision_result.contacts) == 1
            first_name, second_name = next(collision_result.contacts)
            print(first_name, second_name)
            point = np.array([x, y, z])
            points[link_to_check].append(point)

    with args.outfilename.open("wb") as outfile:
        pickle.dump(points, outfile)


if __name__ == "__main__":
    main()
