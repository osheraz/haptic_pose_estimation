#!/usr/bin/env python

'''
Author: Avishai Sintov
Updated by Osher Azulay
'''

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool, Float32
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_mod.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle


class rollout_actor():
    # a node that apply the required actions and publish fails
    # triggers by a flag " rollout_actor/running
    states = []
    plot_num = 0
    reset_status = ' '
    drop = False
    running = False
    action = np.array([0.,0.])
    suc = True
    drop_counter = 0
    max_drop = 3

    def __init__(self):

        rospy.init_node('rollout_actor', anonymous=True)

        rospy.Service('/rollout/run_actor', SetBool, self.callbackTrigger)

        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        # self.arm_reset_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)

        rospy.Subscriber('/hand_control/drop', Bool, self.callbackObjectDrop)
        rospy.Subscriber('/rollout/action', Float32MultiArray, self.callbackAction)
        fail_pub = rospy.Publisher('/rollout/fail', Bool, queue_size = 10) # combine with fail_reason to new type msg
        self.running_pub = rospy.Publisher('/rollout_actor/running', Bool, queue_size = 10)
        self.fail_reason = rospy.Publisher('/rollout_actor/fail_reason', String, queue_size = 10)

        print('[rollout_actor] Ready to rollout...')
        self.running_pub.publish(False)

        self.rate = rospy.Rate(15) # 15hz

        while not rospy.is_shutdown():
            if self.running:

                self.suc = self.move_srv(self.action).success

                if not self.suc:
                    print("[rollout_actor] Load Fail")
                    self.running = False
                    self.running_pub.publish(False)
                    self.fail_reason.publish('load')

                elif self.drop:
                    print("[rollout_actor] Drop Fail")
                    self.running = False
                    self.running_pub.publish(False)
                    self.fail_reason.publish('drop')

                fail_pub.publish(not self.suc or self.drop)

            self.rate.sleep()

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackObjectDrop(self, msg):

        if msg.data:
            self.drop_counter += 1
        else:
            self.drop_counter = 0

        if msg.data and self.drop_counter >= self.max_drop:
            self.drop = True
        else:
            self.drop = False


    def callbackTrigger(self, msg):
        self.running = msg.data
        if self.running:
            self.action *= 0
            print("[rollout_actor] Started ...")
            self.suc = True
            self.running_pub.publish(True)
            self.fail_reason.publish('ok')
        else:
            print("[rollout_actor] Off ...")

        return {'success': True, 'message': ''}


if __name__ == '__main__':
    try:
        rollout_actor()
    except rospy.ROSInterruptException:
        pass