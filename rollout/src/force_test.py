#!/usr/bin/python

'''
Author: Osher Azulay
'''

import rospy
import numpy as np
from std_msgs.msg import Float32
from openhand_node.srv import MoveServos, ReadTemperature

import tf


class Force_test():

    finger_force = np.array([0.])
    normal_force = np.array([0.])


    def __init__(self):

        rospy.init_node('force_test', anonymous=True)

        # Gripper related
        rospy.Subscriber('/finger/force', Float32, self.callbackGripperForce)
        rospy.Subscriber('/normal_force/force', Float32, self.callbackNormalForce)

        self.rate = rospy.Rate(100)

        check_force = 50.0
        eps = 1
        finger_force_list = normal_force_list = []

        while not rospy.is_shutdown():

            if abs(self.normal_force - check_force) < eps:
                finger_force_list.append(self.finger_force)
                normal_force_list.append(self.normal_force)
                print ('Finger force: ' + str(self.finger_force))
                print ('Normal force: ' + str(self.normal_force))

            self.rate.sleep()

    def callbackGripperForce(self, msg):
        self.finger_force = np.array(msg.data)

    def callbackNormalForce(self, msg):
        self.normal_force = np.array(msg.data)



if __name__ == '__main__':

    try:
        Force_test()
    except rospy.ROSInterruptException:
        pass
