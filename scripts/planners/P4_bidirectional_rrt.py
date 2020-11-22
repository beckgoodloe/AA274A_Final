import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection

# Represents a motion planning problem to be solved using the RRT algorithm
class RRTConnect(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

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
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRTConnect")

    def find_nearest_forward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering forward from x
        """
        raise NotImplementedError("find_nearest_forward must be overriden by a subclass of RRTConnect")

    def find_nearest_backward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from x to V[i] is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering backward from x
        """
        raise NotImplementedError("find_nearest_backward must be overriden by a subclass of RRTConnect")

    def steer_towards_forward(self, x1, x2, eps):
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
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRTConnect")

    def steer_towards_backward(self, x1, x2, eps):
        """
        Steers backward from x2 towards x1 along the shortest path (subject
        to robot dynamics). Returns x1 if the length of this shortest path is
        less than eps, otherwise returns the point at distance eps along the
        path backward from x2 to x1.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards_backward must be overriden by a subclass of RRTConnect")

    def solve(self, eps, max_iters=1000):
        """
        Uses RRT-Connect to perform bidirectional RRT, with a forward tree
        rooted at self.x_init and a backward tree rooted at self.x_goal, with
        the aim of producing a dynamically-feasible and obstacle-free trajectory
        from self.x_init to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
                
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

        state_dim = len(self.x_init)

        V_fw = np.zeros((max_iters, state_dim))     # Forward tree
        V_bw = np.zeros((max_iters, state_dim))     # Backward tree

        n_fw = 1    # the current size of the forward tree
        n_bw = 1    # the current size of the backward tree

        P_fw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the forward tree
        P_bw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the backward tree

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V_fw, V_bw, P_fw, P_bw, n_fw, n_bw: the represention of the
        #           planning trees
        #    - success: whether or not you've found a solution within max_iters
        #           RRT-Connect iterations
        #    - self.path: if success is True, then must contain list of states
        #           (tree nodes) [x_init, ..., x_goal] such that the global
        #           trajectory made by linking steering trajectories connecting
        #           the states in order is obstacle-free.
        # Hint: Use your implementation of RRT as a reference

        ########## Code starts here ##########
        V_fw[0, :] = self.x_init
        V_bw[0, :] = self. x_goal
        while(not success):
            x_rand = (np.random.uniform(low=self.statespace_lo[0], high=self.statespace_hi[0], size=1)[0], np.random.uniform(low=self.statespace_lo[1], high=self.statespace_hi[1], size=1)[0], np.random.uniform(low=0, high=1*np.pi, size=1)[0])
            x_near_index = self.find_nearest_forward(V_fw[:n_fw, :], x_rand)
            x_near = V_fw[x_near_index]
            x_new = self.steer_towards_forward(x_near, x_rand, eps)

            # Correct size for state dimension, forethought for dubins
            if(state_dim < 3):
                x_new = [x_new[0], x_new[1]]

            # print("x_near is {} x_new is{}".format(x_near, x_new))
            if(self.is_free_motion(self.obstacles, x_near, x_new)):
                V_fw[n_fw, :] = x_new
                P_fw[n_fw] = x_near_index
                n_fw += 1
                if(n_fw >= max_iters):
                    return False

                x_connect_index = self.find_nearest_backward(V_bw[:n_bw, :], x_new)
                x_connect = V_bw[x_connect_index]
                while True and not success:
                    x_new_connect = self.steer_towards_backward(x_new, x_connect, eps)
                    if(self.is_free_motion(self.obstacles, x_new_connect, x_connect)):
                        V_bw[n_bw, :] = x_new_connect
                        P_bw[n_bw] = x_connect_index
                        n_bw += 1
                        if(n_bw >= max_iters):
                            return False

                        # END CONDITION CHECK
                        if(list(x_new_connect) == list(x_new)):
                            parent = n_fw - 1
                            self.path = []
                            while(parent != -1):
                                self.path.append(V_fw[parent, :])
                                parent = P_fw[parent]
                            self.path = self.path[::-1]

                            parent = n_bw - 1
                            while(parent != -1):
                                self.path.append(V_bw[parent, :])
                                parent = P_bw[parent]

                            success = True
                            break

                        x_connect = x_new_connect
                    else:
                        break

            x_rand = (np.random.uniform(low=self.statespace_lo[0], high=self.statespace_hi[0], size=1)[0], np.random.uniform(low=self.statespace_lo[1], high=self.statespace_hi[1], size=1)[0], np.random.uniform(low=0, high=1*np.pi, size=1)[0])
            x_near_index = self.find_nearest_backward(V_bw[:n_bw, :], x_rand)
            x_near = V_bw[x_near_index]
            x_new = self.steer_towards_backward(x_rand, x_near, eps)

            # Correct size for state dimension, forethought for dubins
            if(state_dim < 3):
                x_new = [x_new[0], x_new[1]]

            if(self.is_free_motion(self.obstacles, x_new, x_near)):
                V_bw[n_bw, :] = x_new
                P_bw[n_bw] = x_near_index
                n_bw += 1
                if(n_bw >= max_iters):
                    return False

                x_connect_index = self.find_nearest_forward(V_fw[:n_fw, :], x_new)
                x_connect = V_fw[x_connect_index]
                while True and not success:
                    x_new_connect = self.steer_towards_forward(x_connect, x_new, eps)
                    if(self.is_free_motion(self.obstacles, x_connect, x_new_connect)):
                        V_fw[n_fw, :] = x_new_connect
                        P_fw[n_fw] = x_connect_index
                        n_fw += 1
                        if(n_fw >= max_iters):
                            return False

                        # END CONDITION CHECK
                        if(list(x_new_connect) == list(x_new)):
                            parent = n_fw - 1
                            self.path = []
                            while(parent != -1):
                                self.path.append(V_fw[parent, :])
                                parent = P_fw[parent]
                            self.path = self.path[::-1]

                            parent = n_bw - 1
                            while(parent != -1):
                                self.path.append(V_bw[parent, :])
                                parent = P_bw[parent]

                            success = True
                            break

                        x_connect = x_new_connect
                    else:
                        break

        # ######### Code ends here ##########

        return success

class GeometricRRTConnect(RRTConnect):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        return np.argmin(np.array([np.sqrt((x[0] - item[0])**2 +
                         (x[1] - item[1])**2) for item in V]))
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        return self.find_nearest_forward(V, x)

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        dist = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        if(dist < eps):
            return x2
        else:
            ratio = eps / dist
            dx = x2[0] - x1[0]
            dy = x2[1] - x1[1]
            return (x1[0] + ratio * dx, x1[1] + ratio * dy)
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        return self.steer_towards_forward(x2, x1, eps)

    def is_free_motion(self, obstacles, x1, x2):
        # Get a number of intermediate points based on the resolution of the grid
        npoints = int(np.ceil(2.0 * np.linalg.norm(np.array(x2) - np.array(x1)) / obstacles.resolution))
        points = [obstacles.snap_to_grid(np.array(x1) + alpha * (np.array(x2) - np.array(x1))) for alpha in np.linspace(0, 1, npoints)]
        # Check that they're all free
        return np.all([obstacles.is_free(point) for point in points]) 

        # x1 = obstacles.snap_to_grid(x1)
        # x2 = obstacles.snap_to_grid(x2)
        # return obstacles.is_free(x1) and obstacles.is_free(x2)

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_tree_backward(self, V, P, **kwargs):
        self.plot_tree(V, P, **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)
