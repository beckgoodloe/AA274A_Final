import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    x_old = []
    y_old = []
    for point in path:
        x_old.append(point[0])
        y_old.append(point[1])

    t_old = [0] * len(path)
    for i in range(1, len(path)):
        dist = ((x_old[i] - x_old[i-1])**2 + (y_old[i] - y_old[i-1])**2)**.5
        t_old[i] = t_old[i-1] + dist / V_des

    t_smoothed = np.arange(0, t_old[-1], dt)

    # Get new splines
    spl_x_old = scipy.interpolate.splrep(t_old, x_old, s=alpha)
    spl_y_old = scipy.interpolate.splrep(t_old, y_old, s=alpha)

    # Evaluate splines at t_smoothed times
    spl_x_new = scipy.interpolate.splev(t_smoothed, spl_x_old)
    spl_y_new = scipy.interpolate.splev(t_smoothed, spl_y_old)

    # Calculate first derivatives
    spl_xd_new = scipy.interpolate.splev(t_smoothed, spl_x_old, 1)
    spl_yd_new = scipy.interpolate.splev(t_smoothed, spl_y_old, 1)

    # Calculate second derivatives
    spl_xdd_new = scipy.interpolate.splev(t_smoothed, spl_x_old, 2)
    spl_ydd_new = scipy.interpolate.splev(t_smoothed, spl_y_old, 2)

    traj_smoothed = np.zeros((len(t_smoothed), 7))
    # Create trajectory
    for t in range(len(t_smoothed)):
        if(t == 0):
            THETA = 0
        else:
            THETA = np.arctan2((spl_y_new[t] - spl_y_new[t-1]),
                               (spl_x_new[t] - spl_x_new[t-1]))
            if(t == 1):
                traj_smoothed[0, 2] = THETA
        traj_smoothed[t] = np.array([spl_x_new[t], spl_y_new[t], THETA, spl_xd_new[t], spl_yd_new[t], spl_xdd_new[t], spl_ydd_new[t]])
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
