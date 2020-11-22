import numpy as np
import math
from utils import wrapToPi
import rospy
from std_msgs.msg import Float64

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1


#rospy.init_node('pose_stabilization_pub_node', anonymous=True)

alpha_pub = rospy.Publisher('/controller/alpha', Float64, queue_size=10)
delta_pub = rospy.Publisher('/controller/delta', Float64, queue_size=10)
rho_pub = rospy.Publisher('/controller/rho', Float64, queue_size=10)

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi (np.unwrap) function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        ro = ((self.y_g - y)**2 + (self.x_g - x)**2)**(.5)
        alpha = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - th)
        delta = wrapToPi(alpha + th - self.th_g)

        rho_pub.publish(ro)
        alpha_pub.publish(alpha)
        delta_pub.publish(delta)

        V = self.k1 * ro * np.cos(alpha)
        if(alpha < ALPHA_THRES):    
            om = self.k2 * alpha + self.k1 * np.cos(alpha) * np.sinc(alpha / math.pi) * (alpha + self.k3 * delta)
        else:
            om = self.k2 * alpha + self.k1 * np.cos(alpha) * np.sin(alpha) / alpha * (alpha + self.k3 * delta)
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
