"""
A_star 2D
@author: huiming zhou
credit: https://github.com/zhm-real/PathPlanning/blob/master/Search_based_Planning/Search_2D/Astar.py
"""

import math
import heapq
from utils import *


#TODO reset() so dont make new occ grid each time

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, heuristic_type, occupancy_grid, resolution=0.2):
        self.heuristic_type = heuristic_type
        # valid movements
        self.res = resolution
        self.occupancy_grid = occupancy_grid
        self.map = occupancy_grid.map
        # print(self.occupancy_grid.map)
        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


    def search(self, s_start, s_goal, cop_positions=None):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.s_start = (int(s_start[0]/self.res), int(s_start[1]/self.res))
        self.s_goal = (int(s_goal[0]/self.res), int(s_goal[1]/self.res))
        if self.s_start not in self.map:
            self.s_start = sorted(self.map, key=lambda x: dist(self.s_start, x))[0]
        if self.s_goal not in self.map:
            self.s_goal = sorted(self.map, key=lambda x: dist(self.s_goal, x))[0]
        # print(self.s_start)
        # print(self.s_goal)
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = np.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n, cop_positions)

                if s_n not in self.g:
                    self.g[s_n] = np.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: np.inf}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = np.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal, cop_positions=None):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return np.inf

        move_cost = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
        if not cop_positions:
            return move_cost
        else:
            cop_positions = [(c[0]/self.res, c[1]/self.res) for c in cop_positions]
            cop_cost = 20/np.mean([math.hypot(s_goal[0] - p[0], s_goal[1] - p[1]) for p in cop_positions])
            return move_cost + cop_cost


    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if not self.occupancy_grid.is_free(s_start) or not self.occupancy_grid.is_free(s_end):
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if not self.occupancy_grid.is_free(s1) or not self.occupancy_grid.is_free(s2):
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return np.array(list(reversed([(p[0]*self.res, p[1]*self.res) for p in path])))

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    s_start = (7, 7)
    s_goal = (-7, -7)
    astar = AStar("euclidean", resolution=0.1, margin=0.5)
    path, visited = astar.search(s_start, s_goal)
    print(np.array(path))

    fig, ax = plt.subplots(figsize=(5,5))
    ax.grid(False)
    ax.imshow(astar.occupancy_grid.as_img())
    pts = [(p[0]/astar.res + astar.occupancy_grid.ticks, p[1]/astar.res + astar.occupancy_grid.ticks) for p in path]
    ax.plot(*zip(*pts))
    plt.show()