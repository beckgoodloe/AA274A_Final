import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    # print("ENTERING COMPUTE DYNAMICS")    
    x, y, theta = xvec
    V, omega = u
    
    theta_new = theta + omega*dt
    if(abs(omega) >= EPSILON_OMEGA):
        x_new = x + V / omega * (np.sin(theta_new) - np.sin(theta))
        y_new = y + V / omega * (np.cos(theta) - np.cos(theta_new))
    else:
        x_new = x + V * np.cos(theta) * dt
        y_new = y + V * np.sin(theta) * dt
        omega = None
    g = np.array([x_new, y_new, theta_new])

    # print(g)
    # print("omega is {} EPSILON_OMEGA is {}".format(omega, EPSILON_OMEGA))
    if compute_jacobians:
        #print("HERE {} {} {} {}".format(V, omega, theta, theta_new))
        Gx = np.zeros((3,3))
        if(omega is not None):        
            dxnew_dth = V / omega * (np.cos(theta_new) - np.cos(theta))
            dynew_dth = V / omega * (np.sin(theta_new) - np.sin(theta))
        else:
            dxnew_dth = -V * np.sin(theta) * dt
            dynew_dth = V * np.cos(theta) * dt
        
        # set diagonals to 1
        for i in range(Gx.shape[0]):
            Gx[i][i] = 1
        Gx[0][2] = dxnew_dth
        Gx[1][2] = dynew_dth

        
        Gu = np.zeros((3,2))
        
        if(omega is not None):
            dxnew_dV = 1 / omega * (np.sin(theta_new) - np.sin(theta))
            dynew_dV = 1 / omega * (np.cos(theta) - np.cos(theta_new))
        else:
            dxnew_dV = np.cos(theta) * dt
            dynew_dV = np.sin(theta) * dt
        dthetanew_dV = 0

        Gu[0][0] = dxnew_dV
        Gu[1][0] = dynew_dV
        Gu[2][0] = dthetanew_dV
        
        #dxnew_dw = (V * ((np.sin(theta_new)*dt - np.sin(theta)))) * (-1/omega**2)
        
        if(omega is not None):
            dxnew_dw=V*((-1/omega**2)*(np.sin(theta_new)-np.sin(theta))+(1./omega)*(np.cos(theta_new)*dt))
            dynew_dw = V*((-1/omega**2)*(np.cos(theta)-np.cos(theta_new))+(1./omega)*(np.sin(theta_new)*dt))
        else:
            dxnew_dw = 0
            dynew_dw = 0
        dthetanew_dw = dt
                
 
        Gu[0][1] = dxnew_dw
        Gu[1][1] = dynew_dw
        Gu[2][1] = dthetanew_dw
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    # unpack
    alpha, r = line
    theta = x[2]
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x_cam, y_cam = x[0:2] + np.dot(rot_matrix, tf_base_to_camera[0:2])
    th_cam = theta + tf_base_to_camera[2]
    al_c = alpha - th_cam
    r_c = r - np.dot(np.array([np.cos(alpha), np.sin(alpha)]), np.array([x_cam, y_cam]))
    h = np.array([al_c, r_c])
    if compute_jacobian:
        da_dx = 0
        da_dy = 0
        da_dth = -1
        dr_dx = -np.cos(alpha)
        dr_dy = -np.sin(alpha)
        x, y = tf_base_to_camera[0:2]
        dr_dth = -np.cos(alpha) * (-x * np.sin(theta) - y * np.cos(theta)) - np.sin(alpha) * (x * np.cos(theta) - y * np.sin(theta))
        Hx = np.array([[da_dx, da_dy, da_dth], [dr_dx, dr_dy, dr_dth]])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
