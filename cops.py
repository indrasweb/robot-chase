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


class Cop:

    __ROBBERS = ['robber_0', 'robber_1', 'robber_2']

    def __init__(self, name, speed=0.6):
        rospy.init_node(name)
        self.name = name
        self.speed = speed
        self.my_pose = GroundtruthPose(name)
        self.occupancy_grid = OccupancyGrid(0.2, 0.5)
        # assign cop to unique robber
        self.target = 'robber_'+name[-1]
        self.target_pose = GroundtruthPose(self.target)
        self.rate_limit = rospy.Rate(100)  # 200ms
        self.path_finder = AStar('euclidean', self.occupancy_grid, resolution=0.2)
        self.velocity_publisher = rospy.Publisher(
            '/{}/cmd_vel'.format(name),
            Twist, queue_size=5
        )
        self.catch_publisher = rospy.Publisher('/caught', String, queue_size=1)
        rospy.Subscriber('/caught', String, self.catch_listener)

        # file for pose history
        self.pose_history = []
        self.pose_history_fn = '/tmp/gazebo_{}.txt'.format(name)
        with open(self.pose_history_fn, 'w'):
            pass

        # wait for ground truth poses node to init
        while not self.my_pose.ready and not self.target_pose.ready:
            self.rate_limit.sleep()
            continue

    @property
    def position(self):
        return (self.my_pose.pose[X], self.my_pose.pose[Y])


    @property
    def target_position(self):
        return (self.target_pose.pose[X], self.target_pose.pose[Y])


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


    def get_path_to_target(self):
        return self.path_finder.search(self.position, self.target_position)[0]


    def check_caught(self):
        if np.linalg.norm(np.array(self.position) - np.array(self.target_position)) < 0.3:
            self.__ROBBERS.remove(self.target)
            print(self.target, 'was caught after', rospy.get_time(), 'seconds by', self.name)
            catch_msg = String()
            catch_msg.data = self.target
            self.catch_publisher.publish(catch_msg)
            if self.__ROBBERS:
                self.target = random.choice(self.__ROBBERS)
                self.target_pose = GroundtruthPose(self.target)

        if self.target not in self.__ROBBERS:
            if self.__ROBBERS:
                self.target = random.choice(self.__ROBBERS)
                self.target_pose = GroundtruthPose(self.target)


    def catch_listener(self, msg):
        if msg.data in self.__ROBBERS:
            self.__ROBBERS.remove(msg.data)


    def check_finished(self):
        if not self.__ROBBERS:
            print('Finished!')
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)
            return True

    def control_loop(self):
        path = self.get_path_to_target()
        s = rospy.get_time()

        while not rospy.is_shutdown():

            self.check_caught()
            if self.check_finished():
                break

            if (rospy.get_time() - s) > 5:  # get new path every 5s
                path = self.get_path_to_target()
                s = rospy.get_time()

            v = get_velocity_to_follow_path(np.array(self.position), path, self.speed)
            u, w = feedback_linearize(self.my_pose.pose, v)
            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w

            self.velocity_publisher.publish(vel_msg)
            self.log_pose()
            self.rate_limit.sleep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', choices=['cop_0', 'cop_1', 'cop_2'])
    args, unknown = parser.parse_known_args()
    try:
        Cop(args.name).control_loop()
    except rospy.ROSInterruptException:
        pass
