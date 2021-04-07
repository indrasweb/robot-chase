import numpy as np
from gazebo_msgs.msg import ModelStates
import rospy
from tf.transformations import euler_from_quaternion

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

ROBOT_RADIUS = 0.105 / 2.
SIZE = 8.
XY_LIMIT = SIZE - ROBOT_RADIUS*2
CYLINDER_POSITIONS = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]], dtype=np.float32)
CYLINDER_RADII = 2. + ROBOT_RADIUS

class OccupancyGrid:

    def __init__(self, resolution=0.2, margin=0.0):
        """ margin : how far away to keep from obstacles """
        self.res = resolution
        self.ticks = int(SIZE / resolution)
        self.map = set((x, y) for x in range(-self.ticks, self.ticks)
                              for y in range(-self.ticks, self.ticks) if self.__is_free(x, y, margin))

    def __is_free(self, x, y, margin=0.0):
        x, y = x * self.res, y * self.res
        x_valid = -XY_LIMIT < x < XY_LIMIT
        y_valid = -XY_LIMIT < y < XY_LIMIT
        dists = [np.sqrt((x - c[X]) ** 2 + (y - c[Y]) ** 2) for c in CYLINDER_POSITIONS]
        cylinder_valid = all(d > CYLINDER_RADII + margin for d in dists)
        return x_valid and y_valid and cylinder_valid

    def unscaled_is_free(self, x, y, margin=0.0):
        x_valid = -XY_LIMIT < x < XY_LIMIT
        y_valid = -XY_LIMIT < y < XY_LIMIT
        dists = [np.sqrt((x - c[X]) ** 2 + (y - c[Y]) ** 2) for c in CYLINDER_POSITIONS]
        cylinder_valid = all(d > CYLINDER_RADII + margin for d in dists)
        return x_valid and y_valid and cylinder_valid

    def is_free(self, point):
        return point in self.map  # O(1) lookup

    def as_img(self):
        as_img = np.zeros((self.ticks*2, self.ticks*2))
        for i in range(-self.ticks, self.ticks-1):
            for j in range(-self.ticks, self.ticks-1):
                if self.__is_free(i, j):
                    as_img[i+self.ticks, j+self.ticks] = 1
        return as_img


class GroundtruthPose(object):

    def __init__(self, name):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name

    def callback(self, msg):
        idx = [i for i, n in enumerate(msg.name) if n == self._name]
        if not idx:
            raise ValueError('Specified name "{}" does not exist.'.format(self._name))
        idx = idx[0]
        self._pose[X] = msg.pose[idx].position.x
        self._pose[Y] = msg.pose[idx].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[idx].orientation.x,
            msg.pose[idx].orientation.y,
            msg.pose[idx].orientation.z,
            msg.pose[idx].orientation.w])
        self._pose[YAW] = yaw

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose


# holonomic control utils

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-2:
        return np.zeros_like(v)
    return v / n

def cap(v, max_speed):
    n = np.linalg.norm(v)
    if n > max_speed:
        return v / n * max_speed
    return v

def get_velocity_to_follow_path(position, path_points, speed=0.1):
    # Stop moving if the goal is reached.
    if len(path_points) == 0 or np.linalg.norm(position - path_points[-1]) < .2:
        return np.zeros_like(position)
    # find the index of the closest path point to the current position
    closest = np.argmin([np.linalg.norm(p - position) for p in path_points])
    # get direction between the robot and the point just ahead of this
    if closest + 1 < len(path_points):
        return cap(normalize(path_points[closest + 1] - position) * speed, speed)
    else:
        return np.zeros_like(position)


def feedback_linearize(pose, velocity, epsilon=0.25):
    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
    w = (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW])) / epsilon
    return u, w

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
