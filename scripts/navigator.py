#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj, GeometricRRT, GeometricRRTConnect
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from asl_turtlebot.msg import DetectedObject
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

from itertools import permutations

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    PICKUP = 4
    STOP = 5
    FREEZE = 6

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE


        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None
        self.replan_flag = False

        self.th_init = 0.0

        # Flag to notify we're in a recovery state
        self.recovering = False

        # Replan only n map updates
        self.map_updates_to_replan = 0
        self.map_updates_since_replan = 0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]

        # stuff regarding pickup
        self.pickup_start = None
        # (3.314, 2.15, 0), (3, 2.1, 1)
        self.queue = []
        self.pickup_time = 3
        self.vendor_list = {'cat':(2.09, 1.74, -.037)}
        self.vendor_dist = dict() 
        
        # flag to clear queue and stop robot
        self.clear_flag = False
        
        # flag to set home
        self.has_run = False

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.7)

        # Time to stop at stop sign
        self.stop_time = 3
        self.crossing = False
        self.cross_time = 10
        
        # Robot limits
        self.v_max = 0.2    # maximum velocity
        self.om_max = 0.4   # maximum angular velocity
        #self.v_max = rospy.get_param("~v_max")
        #self.om_max = rospy.get_param("~om_max")
        print('read in :', self.v_max)
        print('read in: ', self.om_max)

        self.v_des = 0.15   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.

        self.initializeControllers()

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vendor_pub = rospy.Publisher('/valid_keys', String)
        self.marker_pub = rospy.Publisher('/vendor_locs', MarkerArray)

  
        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/new_waypoint", String, self.command_callback)
        rospy.Subscriber("/detector/stop_sign",DetectedObject, self.detected_callback)
        rospy.Subscriber("/detector/apple",DetectedObject, self.detected_callback)
        rospy.Subscriber("/detector/pizza",DetectedObject, self.detected_callback)
        rospy.Subscriber("/detector/broccoli",DetectedObject, self.detected_callback)
        rospy.Subscriber("/detector/banana",DetectedObject, self.detected_callback)
        rospy.Subscriber("/detector/cat",DetectedObject, self.meow_meow)
        rospy.Subscriber("/optimize", String, self.optimize_callback)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    def initializeControllers(self):
        self.pose_controller = PoseController(0., 0., 0., self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)
        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)

    def resetControllerSpeed(self):
        self.pose_controller.V_max = self.v_max
        self.pose_controller.om_max = self.om_max
        
        self.heading_controller.om_max = self.om_max

        self.traj_controller.V_max = self.v_max
        self.traj_controller.om_max = self.om_max

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj


    def publish_markers(self):
        markerArray = MarkerArray()
        num = 1
        for vendor in self.vendor_list.keys():
            if (vendor != "stop_sign") and (vendor != "cat"):
                loc = self.vendor_list[vendor]
                marker = Marker()
                marker.header.frame_id = "/map"
                marker.id = num
                num += 1
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 1.0
                marker.color.a = 1.0
                marker.pose.position.x = loc[0]
                marker.pose.position.y = loc[1] 
                marker.pose.position.z = .1 
                markerArray.markers.append(marker)
        self.marker_pub.publish(markerArray)
    
    def meow_meow(self, data):
        print("OH A CAT MEOW MEOW")

    def optimize_callback(self, data):
        if not len(self.queue) > 0:
            return
        self.switch_mode(Mode.FREEZE)        
        all_dest = list(set(self.queue + [(self.x_g, self.y_g, self.theta_g)]))
        perms = list(permutations(all_dest))
        min_dist = float('inf')
        min_order = None
        first_fail = False
        for path in perms:
            total_dist = 0
            # calculate distance to first point
            # Attempt to plan a path
            state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
            state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
            x_init = self.snap_to_grid((self.x, self.y))
            self.plan_start = x_init
            x_goal = self.snap_to_grid((path[0][0], path[0][1]))
            problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

            # was the planning successful?
            success =  problem.solve()
            # if not successful, do not consider this permutation
            if not success:
                print("FIRST PATH NOT VALID", path)
                first_fail = True
            # if successful, add to total path length for permutation and continue
            else:            
                total_dist += len(problem.path)

            for i in range(len(path)-1):
                if first_fail:
                    first_fail = False
                    break
                # Attempt to plan a path
                state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
                state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
                x_init = self.snap_to_grid((path[i][0], path[i][1]))
                self.plan_start = x_init
                x_goal = self.snap_to_grid((path[i+1][0], path[i+1][1]))
                problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

                # was the planning successful?
                success =  problem.solve()
                # if not successful, do not consider this permutation
                if not success:
                    print("THIS PATH NOT VALID", path)
                    break
                # if successful, add to total path length for permutation and continue
                total_dist += len(problem.path)
            if total_dist < min_dist:
                min_dist = total_dist
                min_order = path
        if min_order is not None:
            print("FOLLOWING NEW PATH", min_order)
            self.queue = list(min_order)
            self.switch_mode(Mode.IDLE)
        else:
            print("FOLLOWING OLD PATH", all_dest)
            self.queue = all_dest
            self.x_g, self.y_g, self.theta_g = self.queue.pop(0)
            self.replan_flag = True
            self.replan()
                
                

    def detected_callback(self,data):
        if data.confidence > 0.5 and data.distance < 1.0:
            print('In detected callback')
            #d_obj = data.distance - 0.3
            #theta_o =0.5*(data.thetaleft + data.thetaright)
            
            #(translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_camera', rospy.Time(0))
            #x_c = translation[0]
            #y_c = translation[1]
            #euler = tf.transformations.euler_from_quaternion(rotation)
            #theta_c = euler[2]
            #x_o = x_c + np.cos(theta_c + theta_o) * d_obj
            #y_o = y_c + np.sin(theta_c + theta_o) * d_obj
            x_o=self.x
            y_o=self.y
            #x_o = self.x + np.cos(self.theta) * d_obj
            #y_o = self.y + np.sin(self.theta) * d_obj
            
            if(data.name == "stop_sign" and not self.crossing and data.confidence > 0.75):
                print("STOP SIGN DISTANCE, Theta", data.distance, data.thetaleft,data.thetaright)
                # distance of the stop sign
                dist = data.distance
                # if close enough and in nav mode, stop
                if dist > 0 and dist < self.stop_min_dist and (self.mode == Mode.TRACK or self.mode == Mode.PARK):
                        self.init_stop_sign()

            if data.name not in self.vendor_list.keys():
                self.vendor_list[data.name] = (x_o, y_o, self.theta)
                self.vendor_pub.publish(String(data.name))
                self.vendor_dist[data.name] = data.distance
                print("ADDING TO VENDORS", x_o, y_o, self.theta, data.name)
                print("Robot coordinates:",self.x,self.y,self.theta)
            else:
                if data.distance < self.vendor_dist[data.name]:
                    print('Updating vendor: ', data.name)
                    self.vendor_list[data.name] = (x_o, y_o, self.theta)
                    self.vendor_dist[data.name] = data.distance
    
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]

        self.v_max = config["v_max"]
        self.om_max = config["om_max"]
        self.v_des = config["v_des"]
        self.resetControllerSpeed()     

        self.plan_resolution = config["resolution"]       
        
        print('config file: ', config)
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        print("CMD CALLBACK")
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan_flag = True
            self.replan()
            print("NEW GOAL PUBLISHED")

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)

            if self.x_g is not None and self.map_updates_since_replan >= self.map_updates_to_replan:
                print('replanning because map updates')
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan_flag = True
                self.replan() # new map, need to replan
                self.map_updates_since_replan = -1 # -1 b/c will +1 to 0 right after
            self.map_updates_since_replan = self.map_updates_since_replan + 1
            #print('Map updates since replan: ', self.map_updates_since_replan)

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        if not self.x_g == None:
            return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh
        return True

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        if self.x_g is not None:
            return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.at_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)
        return True

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct s loaded
        """
        t = self.get_current_plan_time()
        # print('Mode: ', self.mode)
        #print('Crossing: ', self.crossing)
        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        else:
            V = 0.
            om = 0.

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we aren't in STOP or Freeze mode
        if self.mode == Mode.STOP or self.mode == Mode.FREEZE:
            return
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem_astar = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)
        problem_rrt = GeometricRRTConnect(state_min,state_max,x_init,x_goal,self.occupancy)

        rospy.loginfo("Navigator: computing navigation plan")
        success_astar =  problem_astar.solve()

        if not success_astar:
            success_rrt = False
            count = 4
            while not success_rrt and count > 0:
                print("RRT COUNT IS ", count)
                success_rrt = problem_rrt.solve(.1)
                count -= 1
            if not success_rrt:
                rospy.loginfo("Planning failed")
                return
            else:
                print("I USED RRT")
                planned_path = problem_rrt.path
        else:
            planned_path = problem_astar.path
        
        rospy.loginfo("Planning Succeeded")

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]
            if(self.replan_flag):
                self.replan_flag = False
            elif t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.switch_mode(Mode.STOP)
    
    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        print("Twist:",vel_g_msg)
        self.cmd_vel_publisher.publish(vel_g_msg)

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def update_crossing(self):
        self.crossing = (rospy.get_rostime() - self.cross_start) <= rospy.Duration.from_sec(self.cross_time) 

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.switch_mode(Mode.TRACK)
        self.crossing = True

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # publish vendor locs marker array
            self.publish_markers()
            
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
                
                if not self.has_run:
                    self.vendor_list["home"] = (self.x, self.y, self.theta)
                    self.has_run = True
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            #print("Current mode is: ", self.mode)
            if self.clear_flag:
                self.clear_flag = False
                # forget about goal:
                self.x_g = None
                self.y_g = None
                self.theta_g = None
                # clear the queue
                self.queue = []
                # go into idle
                self.switch_mode(Mode.IDLE)
            if self.recovering:
                start_recovery = rospy.get_rostime()
                while rospy.get_rostime() - start_recovery < rospy.Duration.from_sec(3):
                    print("RECOVERING")
                    cmd_vel = Twist()
                    cmd_vel.linear.x = -0.1
                    self.nav_vel_pub.publish(cmd_vel)
                # forget about goal
                self.x_g = None
                self.y_g = None
                self.theta_g = None
                # clear the queue                
                self.queue = []
                # command zero vel              
                cmd_vel = Twist()
                self.nav_vel_pub.publish(cmd_vel)
                # force into IDLE
                self.switch_mode(Mode.IDLE)
                # unflag recovery
                self.recovering = False
            if self.crossing:            
                self.update_crossing()
            if self.mode == Mode.IDLE:
                if len(self.queue) > 0:
                    self.x_g, self.y_g, self.theta_g = self.queue.pop(0)
                    self.replan_flag = True
                    self.replan()
            elif self.mode == Mode.PICKUP:
                # if timer elapsed 
                if(rospy.get_rostime() > self.pickup_start + rospy.Duration.from_sec(self.pickup_time)):
                    # added a second check to make sure queue is not empty due to previous errors
                    if len(self.queue) > 0:
		                # pop first element of queue and publish to cmd_nav
		                self.x_g, self.y_g, self.theta_g = self.queue.pop(0)
		                print("NEW WAYPOINT", self.x_g, self.y_g, self.theta_g)
		                self.replan_flag = True
		                self.replan()
                    else:
                        self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                    rospy.loginfo("finished replanning")
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.STOP:
                print("WE ARE IN STOP MODE")
                self.stay_idle()
                if self.has_stopped():
                    self.init_crossing()
                # At a stop
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    # Check queue and either plan to next waypoint or home
                    if self.queue == []:
                        self.switch_mode(Mode.IDLE)
                    else:
                        # start timer
                        self.pickup_start = rospy.get_rostime()
                        self.switch_mode(Mode.PICKUP)
            elif self.mode == Mode.FREEZE:
                self.stay_idle()


            self.publish_control()
            rate.sleep()

    def command_callback(self, data):
        if data.data == "explore":
            # 0.4853, 2.677, -2.2776
            explore_path = [(2.428, 2.787, 3.14159), (.377,1.603,-3.0), (0.354, 0.373, -.015),(3.394,0.438,1.569)]
            # self.queue.append(explore_path)
            self.queue = self.queue + explore_path
            # print("QUEUE:", self.queue)
        elif data.data == "clear":
            self.clear_flag = True
        elif data.data == "faster":
            self.v_max = self.v_max * 1.33
            self.om_max = self.om_max * 1.33
            self.v_des = self.v_des * 1.33
            self.resetControllerSpeed()
        elif data.data == "slower":
            self.v_max = self.v_max / 1.33
            self.om_max = self.om_max / 1.33
            self.v_des = self.v_des / 1.33
            self.resetControllerSpeed()
        elif data.data == "recover":
            self.recovering = True
        else:
            print(self.vendor_list[data.data])
            self.queue.append(self.vendor_list[data.data])

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
