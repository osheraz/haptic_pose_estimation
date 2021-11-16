#!/usr/bin/env python

'''
Updated by Osher Azulay
'''
import math
import rospy
from past.builtins import raw_input
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout.srv import rolloutReqMod, rolloutReqFile, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf import TransformListener
from openhand_node.srv import MoveServos, ReadTemperature

class Calibrate():

    obj_pos = np.array([0., 0.])
    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    avg_load = np.array([0., 0.])
    load_concat = np.array([0. , 0.])
    finger1 = np.array([0., 0., 0.])
    finger2 = np.array([0., 0., 0.])

    suc = False
    reset_status = 'waiting'
    base_pos = [0,0]
    base_theta = 0

    def __init__(self):
        rospy.init_node('calibration', anonymous=True)
        self.tl = TransformListener()

        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)
        rospy.Subscriber('/rollout/ObjectIsReset', String, self.callbackTrigger)

        rospy.Subscriber('/marker_tracker/rigid_bodies/world/pose', PoseStamped, self.callbackWorldMarkers)
        self.move_servos_srv = rospy.ServiceProxy('/openhand_node/move_servos', MoveServos)

        # gripper services
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.gets_srv = rospy.ServiceProxy('/rollout/get_states', gets)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)
        self.regrasp_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)


        print('[calibrate] Begins calibration procedure ...')

        self.rate = rospy.Rate(10)  # 15hz

        # while not rospy.is_shutdown():

        self.start_calibrate()

            # self.rate.sleep()


    def start_calibrate(self):

        finish = False
        count = 0

        rospy.sleep(1)

        self.trigger = False

        params = np.array([rospy.get_param('/openhandNode/motor_offset')[0],
                           rospy.get_param('/openhandNode/motor_offset')[1],
                           rospy.get_param('/hand_control/finger_close_load')])

        while not finish:

            count += 1
            np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

            self.load_concat = np.vstack((self.load_concat,self.gripper_load)) # load is super noisy.

            if self.load_concat.shape[0] > 200:

                self.avg_load = np.mean(self.load_concat[-100:], axis=0)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')
                # print('finger1 :  ' + str(self.finger1) + '   finger2 :  ' + str(self.finger2))
                # print('average load:  ' + str(self.avg_load) + '   diff:  ' + str(np.array([self.avg_load[0] - self.avg_load[1]])))
                # print('motor angle:  ' + str(self.finger1[1]) + '   diff:  ' + str(np.array([self.gripper_pos[0] - self.gripper_pos[1]])))
                print('1 angle:  ' + str(self.finger1[2]) + '    2 angle:  ' + str(self.finger2[2]) )

                y_init = 0.1
                angle_init = 50
                # figner [ x , y , q ]
                delta_y = np.array([y_init - self.finger2[1], y_init - self.finger1[1]])  # desired y init
                delta_a = np.array([angle_init - self.finger2[2], angle_init - self.finger1[2]])  # desired angle init

                # print('Diff motor angle to desire:  ' + str(delta_a))
                # print('Diff motor y to desire:  ' + str(delta_y))

                delta_a_action = np.clip(delta_a, -0.01, 0.01)
                print('action to apply ' + str(delta_a_action))
                self.suc = self.move_srv(delta_a_action).success

                if np.linalg.norm(delta_a) < 0.1:
                    finish = True

                self.rate.sleep()

        p = np.array([self.gripper_pos[0], self.gripper_pos[1]])

        print('Offset:  ' + str(p))

        print('please add 1st - motor offset ' + str(-p[0]))
        print('please add 2nd - motor offset ' + str(p[1]))


        rospy.sleep(1.)
        # self.open_srv()

    def reset(self):

        print("[calibrate] Regrasping the object")
        while 1:
            if not self.trigger:
                print('[calibrate] Waiting for arm to grasp object...')
                self.regrasp_srv()
                rospy.sleep(1.0)
            self.rate.sleep()
            if self.reset_status != 'moving' and self.trigger:
                self.rate.sleep()
                if self.drop_srv().dropped:  # Check if really grasped
                    self.trigger = False
                    print('[calibrate] Grasp failed. Restarting')
                    continue
                else:
                    break
        self.trigger = False

    def callbackTrigger(self, msg):
        self.reset_status = msg.data
        if not self.trigger and self.reset_status == 'finished':
            self.trigger = True

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.abs(np.array(msg.data))

    def callbackGripperPos(self, msg):
        self.gripper_pos = np.array(msg.data)

    def callbackWorldMarkers(self, msg):
        try:
            quaternion = (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w)
            roll,pitch,yaw = euler_from_quaternion(quaternion)
            self.base_theta = np.array(yaw)
            self.base_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        except:
            self.base_pos = np.array([np.nan, np.nan])
            self.base_theta = np.nan

        (trans1, rot1) = self.tl.lookupTransform("world", "finger1", rospy.Time(0))
        (trans2, rot2) = self.tl.lookupTransform("world", "finger2", rospy.Time(0))

        try:
            roll1,pitch1,yaw1 = euler_from_quaternion(rot1)
            if yaw1 > np.pi: yaw1 -= 2. * np.pi
            if yaw1 < -np.pi: yaw1 += 2. * np.pi
            yaw1 = np.arctan2(abs(trans1[0]), abs(trans1[1]))
            self.finger1[2] = abs(yaw1*180/np.pi)
            self.finger1[:2] = np.array([trans1[1], trans1[0]])
        except:
            self.finger1 = np.array([np.nan, np.nan, np.nan])

        try:
            roll2,pitch2,yaw2 = euler_from_quaternion(rot2)
            if yaw2 > np.pi: yaw2 -= 2. * np.pi
            if yaw2 < -np.pi: yaw2 += 2. * np.pi
            yaw2 = np.arctan2(abs(trans2[0]), abs(trans2[1]))
            self.finger2[2] = yaw2*180/np.pi
            self.finger2[:2] = np.array([trans2[1], trans2[0]])
        except:
            self.finger2 = np.array([np.nan, np.nan, np.nan])


if __name__ == '__main__':
    try:
        Calibrate()
    except rospy.ROSInterruptException:
        pass
