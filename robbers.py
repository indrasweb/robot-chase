#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy
from utils import *
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from astar import AStar
import random
import time


class Robber:

    def __init__(self, name, speed=0.8):
        rospy.init_node(name)
        self.name = name
        self.speed = speed
        self.caught = False
        self.my_pose = GroundtruthPose(name)
        self.cop_poses = [GroundtruthPose(c) for c in ['cop_0', 'cop_1', 'cop_2']]
        self.rate_limit = rospy.Rate(100)  # 200ms
        self.res = 0.2
        self.occupancy_grid = OccupancyGrid(self.res, 0.5)
        self.path_finder = AStar('euclidean', self.occupancy_grid, resolution=self.res)
        self.map = list(self.path_finder.occupancy_grid.map)
        self.velocity_publisher = rospy.Publisher(
            '/{}/cmd_vel'.format(name),
            Twist, queue_size=5
        )
        rospy.Subscriber('/caught', String, self.catch_listener)

        # file for pose history
        self.pose_history = []
        self.pose_history_fn = '/tmp/gazebo_{}.txt'.format(name)
        with open(self.pose_history_fn, 'w'):
            pass

        # wait for ground truth poses node to init
        while not self.my_pose.ready or not all(c.ready for c in self.cop_poses):
            self.rate_limit.sleep()
            continue

        self.target_position, self.path = self.get_random_target_position_and_path()

    @property
    def position(self):
        return (self.my_pose.pose[X], self.my_pose.pose[Y])


    def braitenberg(self, f, fl, fr, l, r):
        f, fl, fr, l, r = np.tanh([f, fl, fr, l, r])
        u = min(self.speed, 0.2*(abs(fl)+abs(fr)+abs(f)))
        w = 0.2*fl + 0.1*l + 0.2*f - 0.2*fr - 0.1*r
        return u, w


    def log_pose(self):
        self.pose_history.append(self.my_pose.pose)
        if len(self.pose_history) % 50:
            with open(self.pose_history_fn, 'a') as fp:
                fp.write('\n'.join(','.join(format(v, '.5f') for v in p) for p in self.pose_history) + '\n')
            self.pose_history = []

    def line_of_sight(self, a, b):
        ax, ay = a[0] / self.occupancy_grid.res, a[1] / self.occupancy_grid.res
        bx, by = b[0] / self.occupancy_grid.res, b[1] / self.occupancy_grid.res
        line_x, line_y = np.arange(ax, bx, 1), np.arange(ay, by, 1)
        return all(self.occupancy_grid.is_free(p) for p in zip(line_x, line_y))


    def get_random_target_position_and_path(self):
        choices = [random.choice(self.map) for _ in range(5)]
        choices = [(c[0]*self.res, c[1]*self.res) for c in choices]
        cops = [c.pose[:2] for c in self.cop_poses if self.line_of_sight(self.my_pose.pose, c.pose)]
        ranked = sorted(choices, key=lambda x: sum(dist(c, x) for c in cops)/len(cops) if cops else 0)
        goal = ranked[-1]
        path = self.path_finder.search(self.position, goal, cops)[0]
        return goal, path

    def get_maximin_random_target_position(self):
        choices = {random.choice(self.map): np.inf for _ in range(5)}
        cops = [c.pose[:2] for c in self.cop_poses if self.line_of_sight(self.my_pose.pose, c.pose)]
        for goal_point in choices.keys():
            minimax = np.inf
            for cop_location in cops:
                cop_location_res = (cop_location[0]*self.res, cop_location[1]*self.res)
                path = self.path_finder.search(cop_location_res, goal_point)[0]
                for path_point in path:
                    d = dist(path_point, cop_location)
                    if d < minimax:
                        minimax = d
            choices[goal_point] = minimax
        best = max(choices, key=choices.get)
        return best


    def check_goal(self):
        return dist(self.position, self.target_position) < 0.2


    def get_path_to_target(self):
        return self.path_finder.search(self.position, self.target_position)[0]


    def catch_listener(self, msg):
        if msg.data == self.name:
            self.caught = True

    def control_loop(self):
        s = rospy.get_time()

        while not rospy.is_shutdown():

            if self.caught:
                vel_msg = Twist()
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)
                break

            if self.check_goal() or rospy.get_time() - s > 10:
                self.target_position, self.path = self.get_random_target_position_and_path()
                s = rospy.get_time()

            v = get_velocity_to_follow_path(np.array(self.position), self.path, self.speed)
            u, w = feedback_linearize(self.my_pose.pose, v)
            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w

            self.velocity_publisher.publish(vel_msg)
            self.log_pose()
            self.rate_limit.sleep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', choices=['robber_0', 'robber_1', 'robber_2'])
    args, unknown = parser.parse_known_args()
    try:
        Robber(args.name).control_loop()
    except rospy.ROSInterruptException:
        pass
