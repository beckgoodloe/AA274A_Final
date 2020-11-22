#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

pose = Pose()

def callback(data):
	# Update the pose
	print(data.pose)
	pose.position = data.pose.position
	pose.orientation = data.pose.orientation

def initSubscriber():
	# rospy.init_node("goal_subscriber", anonymous=True)
	rospy.Subscriber("/move_base_simple/goal", PoseStamped, callback)
	# rospy.spin()

def initPublisher():
	vis_pub = rospy.Publisher('goal_location', Marker, queue_size=10)
	rospy.init_node('goal_node', anonymous=True)
	return vis_pub

def publisher(our_pub):
	marker = Marker()

	marker.header.frame_id = "map"
	marker.header.stamp = rospy.Time()

	# IMPORTANT: If you're creating multiple markers, 
	#			each need to have a separate marker ID.
	marker.id = 0

	marker.type = 0 # sphere

	marker.pose.position.x = pose.position.x
	marker.pose.position.y = pose.position.y
	marker.pose.position.z = pose.position.z

	marker.pose.orientation.x = pose.orientation.x
	marker.pose.orientation.y = pose.orientation.y
	marker.pose.orientation.z = pose.orientation.z
	marker.pose.orientation.w = pose.orientation.w

	marker.scale.x = 1.0
	marker.scale.y = 0.1
	marker.scale.z = 0.1

	marker.color.a = 1.0 # Don't forget to set the alpha!
	marker.color.r = 1.0
	marker.color.g = 0.0
	marker.color.b = 0.0
	
	our_pub.publish(marker)	
		
	# print('Published marker!')


if __name__ == '__main__':
	try:
		our_pub = initPublisher()
		initSubscriber()
		rate = rospy.Rate(1)
		while not rospy.is_shutdown():
			publisher(our_pub)
			rate.sleep()
			
	except rospy.ROSInterruptException:
		pass
