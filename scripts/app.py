#!/usr/bin/env python
import rospy
from std_msgs.msg import String

valid_keys = ["home", "explore", "clear", "slower", "faster", "cat", "recover"]

def callback(data):
	valid_keys.append(data.data)
	print("NEW KEY ADDED: ", data.data)

def initSubscriber():
	rospy.Subscriber("/valid_keys", String, callback)

def initPublisher():
	waypoint_pub = rospy.Publisher('new_waypoint', String, queue_size=10)
	rospy.init_node('new_waypoint', anonymous=True)
	return waypoint_pub

def initOptimizePublisher():
	pub = rospy.Publisher('optimize', String, queue_size=10)
	# rospy.init_node('optimize', anonymous=True)
	return pub

def publisher(our_pub, waypoint):
	our_pub.publish(waypoint)

if __name__ == '__main__':
	try:
		waypoint_update_pub = initPublisher()
		optimize_pub = initOptimizePublisher()
		initSubscriber()
		rate = rospy.Rate(1)
		while not rospy.is_shutdown():
			value = raw_input("Enter new command: ")
			if value == "optimize":
				optimize_pub.publish(value)			
			elif value in valid_keys:
				publisher(waypoint_update_pub, value)
			elif value == "keys":
				print(valid_keys)			
			else:
				print("Not a valid command")
	except rospy.ROSInterruptException:
		pass
