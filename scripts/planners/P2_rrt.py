import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection
import matplotlib.patches as patches
import rospy

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = []       # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    def solve(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters, state_dim))
        V[0, :] = self.x_init    # RRT is rooted at self.x_init
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters, dtype=int)

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        ## Hints:
        #   - use the helper functions find_nearest, steer_towards, and is_free_motion
        #   - remember that V and P always contain max_iters elements, but only the first n
        #     are meaningful! keep this in mind when using the helper functions!

        ########## Code starts here ##########
        x_last = []
        while(n < max_iters and not success):
            # Implement goal bias
            z = np.random.rand()
            if(z < goal_bias):
                x_rand = self.x_goal
            else:
                x_rand = (np.random.uniform(low=self.statespace_lo[0], high=self.statespace_hi[0], size=1)[0], np.random.uniform(low=self.statespace_lo[1], high=self.statespace_hi[1], size=1)[0], np.random.uniform(low=0, high=1*np.pi, size=1)[0])
            # Find nearest point and index
            x_near_index = self.find_nearest(V[:n, :], x_rand)
            x_near = V[x_near_index, :]

            x_last = x_near

            # Calculate new point based on line between near and rand
            x_new = self.steer_towards(x_near, x_rand, eps)

            if(self.is_free_motion(self.obstacles, x_near, x_new)):
                V[n, :] = x_new
                P[n] = x_near_index
                # if(x_new[0] == self.x_goal[0] and x_new[1] == self.x_goal[1]):
                if(list(x_new) == list(self.x_goal)):
                    parent = n
                    self.path = []
                    while(parent != -1):
                        self.path.append(V[parent, :])
                        parent = P[parent]
                    self.path = self.path[::-1]
                    success = True
                n += 1
        # ######### Code ends here ##########

        return success

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        success = False
        while(not success):
            success = True
            n = 1
            while(n < len(self.path) - 1):
                x = self.path[n]
                # if((x[0] == self.x_init[0] and x[1] == self.x_init[1]) or (x[0] == self.x_goal[0] and x[1] == self.x_goal[1])):
                #     n += 1
                #     continue
                if(self.is_free_motion(self.obstacles, self.path[n-1], self.path[n+1])):
                    self.path.pop(n)
                    success = False
                else:
                    n += 1
        ########## Code ends here ##########

class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        r, c = np.shape(V)
        x = (x[0], x[1])
        nearest = None
        min_dist = float('inf')
        for j in range(r):
            current_dist = np.sqrt((V[j][0] - x[0])**2 + (V[j][1] - x[1])**2)
            if(current_dist < min_dist):
                min_dist = current_dist
                nearest = j
        return nearest
        # ######### Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        # ######### Code starts here ##########
        # Hint: This should take one line.
        x2 = (x2[0], x2[1])
        dist = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        if(dist < eps):
            return x2
        else:
            ratio = eps / dist
            dx = x2[0] - x1[0]
            dy = x2[1] - x1[1]
            return (x1[0] + ratio * dx, x1[1] + ratio * dy)
        # ######### Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2):
        x1 = obstacles.snap_to_grid(x1)
        x2 = obstacles.snap_to_grid(x2)
        return obstacles.is_free(x1) and obstacles.is_free(x2)
