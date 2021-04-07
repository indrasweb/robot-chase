from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

ROBOT_RADIUS = 0.105 / 2.
SIZE = 8.
XY_LIMIT = SIZE - ROBOT_RADIUS
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

  def is_free(self, point):
    return point in self.map  # O(1) lookup

  def as_img(self):
    as_img = np.zeros((self.ticks * 2, self.ticks * 2))
    for i in range(-self.ticks, self.ticks - 1):
      for j in range(-self.ticks, self.ticks - 1):
        if self.__is_free(i, j):
          as_img[i + self.ticks, j + self.ticks] = 1
    return as_img

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', action='store')
  parser.add_argument('--resolution', default=0.1, action='store')
  parser.add_argument('--margin', default=0.5, action='store')
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots(figsize=(5, 5))
  grid = OccupancyGrid(args.resolution, margin=args.margin)
  ax.grid(False)
  ax.imshow(grid.as_img())

  fns = os.listdir(args.dir)
  for fn in fns:
    if fn.endswith('txt') and 'cop' in fn:
      data = np.genfromtxt(f'{args.dir}/{fn}', delimiter=',')
      pts = [(p[0]/args.resolution + grid.ticks, p[1]/args.resolution + grid.ticks) for p in data]
      ax.plot(*zip(*pts), color='green')
      ax.add_patch(plt.Circle(pts[0], 2, color='green', alpha=0.8))

  fns = os.listdir(args.dir)
  for fn in fns:
    if fn.endswith('txt') and 'robber' in fn:
      data = np.genfromtxt(f'{args.dir}/{fn}', delimiter=',')
      pts = [(p[0] / args.resolution + grid.ticks, p[1] / args.resolution + grid.ticks) for p in data]
      ax.plot(*zip(*pts), color='red')
      ax.add_patch(plt.Circle(pts[0], 2, color='red', alpha=0.8))
      ax.add_patch(plt.Circle(pts[-1], 3, color='blue', alpha=0.8))

  plt.show()