#!/usr/bin/env python

'''
Author: Avishai Sintov
Updated by Osher Azulay
'''

import rospy
from past.builtins import raw_input
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout.srv import rolloutReqMod, rolloutReqFile, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close, TargetPos
import numpy as np
import matplotlib.pyplot as plt
import pickle


class rolloutPublisher():
    # launches the rollout nodes which control the recorder and actor
    # with 3 services
    # one for rollout from actions sequences, second from file and a stop trigger
    obj_pos = np.array([0., 0.])
    reason = ''
    reset_status = 'waiting'
    trigger = False  # Enable collection
    drop = suc = True
    S, A, T, NS, TP = [], [], [], [], []
    start_rot, start_rot_obj, T0 = 0., 0., 0.
    drop_counter = 0

    def __init__(self):
        rospy.init_node('rollout', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReqMod, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/run_trigger', SetBool, self.callbackStop)  # stop trigger

        self.action_pub = rospy.Publisher('/rollout/action', Float32MultiArray, queue_size=10)  # publish the action
        self.fail_reason = rospy.Publisher('/rollout/fail_reason', String, queue_size = 10)

        rospy.Subscriber('/rollout/ObjectIsReset', String, self.callbackTrigger)
        rospy.Subscriber('/hand_control/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/hand_control/drop', Bool, self.callbackObjectDrop)
        self.record_srv = rospy.ServiceProxy('/recorder/record_trigger', SetBool)  # triggers the recorder

        # gripper services
        self.regrasp_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.gets_srv = rospy.ServiceProxy('/rollout/get_states', gets)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)
        self.target_pos_srv = rospy.ServiceProxy('/TargetPos', TargetPos)

        self.state_dim = 9  # TODO: change to get from parameter server..
        self.action_dim = 2
        self.stepSize = 1

        print('[rollout] Ready to rollout...')

        self.rate = rospy.Rate(10)  # 15hz
        rospy.spin()


    def run_rollout(self, A):

        self.trigger = False
        self.reset()
        self.fail = False

        # print("[rollout] Place object and press key...")
        # raw_input()
        # self.close_srv()

        msg = Float32MultiArray()

        print("[rollout] Rolling-out actions...")

        self.record_srv(True)

        # Publish episode actions
        self.running = True
        success = True
        step_counter = 0
        i = 0

        state = np.array(self.obs_srv().state)

        if not self.start_rot:
            self.start_rot = 0 * state[2]
            self.start_rot_obj = 0 * state[2] % (np.pi / 2.) # 4 for sqr, without for cyl

        state[2] = state[2] - self.start_rot + self.start_rot_obj

        if state[2] > np.pi: state[2] -= 2. * np.pi
        if state[2] < -np.pi: state[2] += 2. * np.pi

        st = rospy.get_time()

        while self.running:

            self.S.append(np.copy(state))

            if step_counter <= 0:

                if i > 400:
                    # action = A[i, :] * np.random.choice(range(0, 2), 2) * 0.1
                    # step_counter = self.stepSize
                    action = np.random.uniform(low=-0.005, high=+0.02, size=2)
                    step_counter = np.random.randint(100)
                    i += 1
                else:
                    action = A[i, :]* 0.1 # * 0.1[::-1]
                    step_counter = self.stepSize
                    # if not i % 20:
                    #     action = np.random.uniform(low=-0.005, high=+0.01, size=2)
                    #     step_counter = np.random.randint(40)
                    i += 1

            print('[rollout] Applying action:  ' + str(action) + '\ttime:\t' + str(rospy.get_time() - st))

            msg.data = action
            self.action_pub.publish(msg)
            self.A.append(np.copy(action))
            self.TP.append(np.array(self.target_pos_srv().angle))
            self.T.append(rospy.get_time() - self.T0)

            self.suc = self.move_srv(action).success

            next_state = np.array(self.obs_srv().state)

            next_state[2] = next_state[2] - self.start_rot + self.start_rot_obj

            if next_state[2] > np.pi: next_state[2] -= 2. * np.pi
            if next_state[2] < -np.pi: next_state[2] += 2. * np.pi

            self.NS.append(np.copy(next_state))

            state = np.copy(next_state)
            step_counter -= 1

            if not self.suc:
                print("[rollout] Load Fail")
                self.running = False
                success = False
                self.fail_reason.publish('load')
                self.reason = 'load'
                break

            elif self.drop:
                print("[rollout] Drop Fail")
                self.running = False
                success = False
                self.fail_reason.publish('drop')
                self.reason = 'drop'
                break

            if i >= A.shape[0] and step_counter == 0:
                print("[rollout] Complete.")
                success = True
                self.reason = 'ok'
                break

            self.rate.sleep()

        self.record_srv(False)

        rospy.sleep(1.)
        self.open_srv()

        return success

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

    def slow_open(self):
        print("Opening slowly.")
        for _ in range(30):
            self.move_srv(np.array([-6., -6.]))
            rospy.sleep(0.1)

    def callbackTrigger(self, msg):
        self.reset_status = msg.data
        if not self.trigger and self.reset_status == 'finished':
            self.trigger = True

    def callbackStop(self, msg):
        self.running = msg.data

        return {'success': True, 'message': ''}

    def CallbackRollout(self, req):

        print('[rollout] Rollout request received.')

        actions = np.array(req.actions).reshape(-1, self.action_dim)

        self.S = []
        self.NS = []
        self.A = []
        self.T = []
        self.TP = []
        self.T0 = rospy.get_time()
        self.start_rot = 0.0
        self.start_rot_obj = 0.0

        success = self.run_rollout(actions)

        states = np.array(self.S)
        next_states = np.array(self.NS)
        actions = np.array(self.A)
        target_pos = np.array(self.TP)
        time = np.array(self.T)

        print('[rollout] Recording stopped with %d points.' % len(self.S))

        return {'states': states.reshape((-1,)), 'actions': actions.reshape((-1,)), 'next_states': next_states.reshape((-1,)),
                'time': time.reshape((-1,)), 'success': success, 'reason': self.reason ,'target_pos': target_pos.reshape((-1,))}

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:, :2]
        success = self.run_rollout(actions)

        states = np.array(self.S)

        return {'states': states.reshape((-1,)), 'success': True}

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)

    def callbackObjectDrop(self, msg):

        if msg.data:
            self.drop_counter += 1
        else:
            self.drop_counter = 0

        if msg.data and self.drop_counter >= 3:
            self.drop = True
        else:
            self.drop = False


if __name__ == '__main__':
    try:
        rolloutPublisher()
    except rospy.ROSInterruptException:
        pass
