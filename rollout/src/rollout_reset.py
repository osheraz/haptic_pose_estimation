#!/usr/bin/env python

'''
Author: Avishai Sintov
Updated by Osher Azulay
'''

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool, Float32
from std_srvs.srv import Empty, EmptyResponse
from openhand_node.srv import MoveServos, ReadTemperature
from hand_control.srv import TargetAngles, IsDropped, observation, close, ObjOrientation, RegraspObject
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from common_msgs_gl.srv import SendDoubleArray, SendBool
import geometry_msgs.msg
import math
import time



class rollout_reset():
    # a node that apply the required actions and publish fails
    # triggers by a flag " rollout_actor/running

    reset_status = ' '
    gripper_status = ' '
    drop = False
    drop_counter = 0
    obj_pos = [0,0]
    obj_height = 0.0

    def __init__(self):

        rospy.init_node('rollout_reset')

        rospy.Service('/RegraspObject', RegraspObject, self.RegraspObject)

        rospy.Subscriber('/hand_control/gripper_status', String, self.gripperStatus)
        rospy.Subscriber('/hand_control/obj_pos', Float32MultiArray, self.callbackObjectPos)
        rospy.Subscriber('/hand_control/object_height', Float32, self.callbackObjectHeight)

        self.pub_reset = rospy.Publisher('/rollout/ObjectIsReset', String, queue_size=10)
        self.pub_grasped = rospy.Publisher('/rollout/ObjectIsGrasped', Bool, queue_size=10)

        self.check_drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)
        self.random_close_srv = rospy.ServiceProxy('/RandomCloseGripper', close)

        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        self.move_reset_servo_srv = rospy.ServiceProxy('/openhand_reset_node/move_torque', MoveServos)  # TODO: change srv name..
        self.move_servos_srv = rospy.ServiceProxy('/openhand_node/move_servos', MoveServos)

        print('[rollout_reset] Ready')

        rospy.spin()

    def RegraspObject(self, msg):
        self.pub_reset.publish("moving")
        grasp = self.regrasp_torque()
        if grasp:
            self.pub_reset.publish("finished")
        else:
            self.pub_reset.publish("waiting")
        return {'success': grasp}

    def regrasp_torque(self):

        if self.gripper_status == 'closed':
            self.move_servos_srv.call([0.0, 0.0])

        rospy.sleep(1)
        self.open_srv()
        rospy.sleep(1)

        self.drop_counter += 1
        print('[rollout_reset] Regrasping Object...')
        self.open_srv()
        rospy.sleep(1)
        for i in range(10):
            if self.obj_height < -0.01: #1
                dir = -1
                self.move_reset_servo_srv.call([dir * 0.030])
                rospy.sleep(0.1)
            elif self.obj_height > 0.02:
                dir = 1
                self.move_reset_servo_srv.call([dir * 0.01])
            else:
                break
            rospy.sleep(0.5)

        try:
            suc = self.close_srv().success
        except:
            print("[rollout_reset] closing fingers error")
            grasp = False
        rospy.sleep(2)
        try:
            self.move_reset_servo_srv.call([0.03])
        except:
            print("[rollout_reset] closing reset error")
            grasp = False
        rospy.sleep(1)

        # Verify..
        print('[rollout_reset] Verifying grasp...')
        dr = self.check_drop_srv().dropped
        if dr:  # Check if really grasped
            grasp = False
            print('[rollout_reset] Grasp failed.')
        else:
            self.drop_counter = 0
            grasp = True
            self.pub_grasped.publish(True)

        return grasp

    def callbackObjectPos(self,msg):
        self.obj_pos = np.array(msg.data)
        # print(self.obj_pos[2])

    def callbackObjectHeight(self,msg):
        self.obj_height = np.array(msg.data)
        # print(self.obj_pos[2])

    def gripperStatus(self,msg):
        self.gripper_status = msg.data



if __name__ == '__main__':
    try:
        rollout_reset()
    except rospy.ROSInterruptException:
        pass