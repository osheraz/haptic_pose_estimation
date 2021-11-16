#!/usr/bin/env python

'''
Author: Avishai Sintov
Updated by Osher Azulay
'''

import rospy
from past.builtins import raw_input
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_mod.srv import rolloutReq, rolloutReqFile, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle


class rolloutPublisher():
    # launches the rollout nodes which control the recorder and actor
    # with 3 services
    # one for rollout from actions sequences, second from file and a stop trigger
    obj_pos = np.array([0., 0.])
    states = []
    reason = ''
    plot_num = 0
    reset_status = 'waiting'
    trigger = False  # Enable collection
    drop = True
    suc = True
    fail = False
    actor_running = False

    def __init__(self):
        rospy.init_node('rollout', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/run_trigger', SetBool, self.callbackStop)  # stop trigger

        self.rollout_actor_srv = rospy.ServiceProxy('/rollout/run_actor', SetBool)  # triggers the actor
        self.record_srv = rospy.ServiceProxy('/rollout/record_trigger', SetBool)  # triggers the recorder
        self.action_pub = rospy.Publisher('/rollout/action', Float32MultiArray, queue_size=10)  # publish the action

        rospy.Subscriber('/rollout/fail', Bool, self.callbacFail)  # in case of fail
        rospy.Subscriber('/rollout_actor/running', Bool, self.callbackActorRunning)
        rospy.Subscriber('/rollout/ObjectIsReset', String, self.callbackTrigger)
        rospy.Subscriber('/rollout_actor/fail_reason', String, self.callbackReason)
        rospy.Subscriber('/hand_control/obj_pos', Float32MultiArray, self.callbackObj)

        # rospy.Subscriber('/hand_control/drop', Bool, self.callbackObjectDrop)

        # gripper services
        self.regrasp_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.gets_srv = rospy.ServiceProxy('/rollout/get_states', gets)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)

        self.state_dim = 12  # TODO: change to get from parameter server..
        self.action_dim = 2
        self.stepSize = 1

        print('[rollout] Ready to rollout...')

        self.rate = rospy.Rate(10)  # 15hz
        rospy.spin()

    def reset(self):
        print("[rollout] Regrasping the object")
        while 1:
            if not self.trigger:
                print('[rollout] Waiting for arm to grasp object...')
                self.regrasp_srv()
                rospy.sleep(1.0)
            self.rate.sleep()
            if self.reset_status != 'moving' and self.trigger:
                self.rate.sleep()
                if self.drop_srv().dropped:  # Check if really grasped
                    self.trigger = False
                    print('[rollout] Grasp failed. Restarting')
                    continue
                else:
                    break
        self.trigger = False

    def run_rollout(self, A):

        self.trigger = False

        self.reset()
        # self.regrasp_srv()

        self.fail = False

        # print("[rollout] Place object and press key...")
        # raw_input()
        # self.close_srv()

        msg = Float32MultiArray()

        state = np.array(self.obs_srv().state)
        self.S = []
        self.S.append(np.copy(state))

        print("[rollout] Rolling-out actions...")
        self.rollout_actor_srv(True)
        self.record_srv(True)

        # Publish episode actions
        self.running = True
        success = True
        step_counter = 0
        i = 0
        while self.running:

            if step_counter == 0:

                if i > 300:
                    action = A[i, :] * np.random.choice(range(-1,2),2) # np.random.uniform(low=-0.15, high=+0.15, size=(1, 2))
                    action = action.squeeze()
                    step_counter = 20
                else:
                    action = A[i, :]
                    step_counter = self.stepSize
                    i += 1
                    # print('[rollout] Applying action (' + str(action) + ') '
                    #       + str(i) + ' out of ' + str(A.shape[0]) + '.')

            print('[rollout] Applying action (' + str(action) + ') ')

            msg.data = action
            self.action_pub.publish(msg) # TODO: fix sync
            step_counter -= 1

            if self.fail:  # not suc or fail:
                success = False
                print("[rollout] Drop\Load Fail.")
                self.record_srv(False)
                self.rollout_actor_srv(False)
                break

            if i == A.shape[0] and step_counter == 0:
                print("[rollout] Complete.")
                success = True
                self.record_srv(False)
                self.rollout_actor_srv(False)
                break

            self.rate.sleep()

        rospy.sleep(1.)
        self.open_srv()

        return success

    def slow_open(self):
        print("Opening slowly.")
        for _ in range(30):
            self.move_srv(np.array([-6., -6.]))
            rospy.sleep(0.1)

    def callbacFail(self, msg):
        self.fail = msg.data

    def callbackTrigger(self, msg):
        self.reset_status = msg.data
        if not self.trigger and self.reset_status == 'finished':
            self.trigger = True

    def callbackReason(self, msg):
        self.reason = msg.data

    def callbackStop(self, msg):
        self.running = msg.data

        return {'success': True, 'message': ''}

    def callbackActorRunning(self, msg):
        self.actor_running = msg.data

    def CallbackRollout(self, req):

        print('[rollout] Rollout request received.')

        actions = np.array(req.actions).reshape(-1, self.action_dim)
        success = self.run_rollout(actions)

        SA = self.gets_srv()
        states = np.array(SA.states)
        actions = np.array(SA.actions)
        time = np.array(SA.time)

        return {'states': states.reshape((-1,)), 'actions': actions.reshape((-1,)),
                'time': time.reshape((-1,)), 'success': success, 'reason': self.reason}

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:, :2]
        success = self.run_rollout(actions)

        states = np.array(self.S)

        return {'states': states.reshape((-1,)), 'success': True}

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)


if __name__ == '__main__':
    try:
        rolloutPublisher()
    except rospy.ROSInterruptException:
        pass
