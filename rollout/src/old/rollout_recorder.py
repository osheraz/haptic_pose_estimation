#!/usr/bin/env python

'''
Author: Avishai Sintov
Updated by Osher Azulay
'''

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool, Float32
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_mod.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
from pose_node.srv import one_predict

import matplotlib.pyplot as plt
import pickle
import geometry_msgs.msg
from sensor_msgs.msg import CompressedImage, Image
import os
from cv_bridge import CvBridge, CvBridgeError

record_images = False

class rolloutRecorder():
    # simple record node
    # require trigger - /rollout/record_trigger
    # launch a service to get state

    drop = True
    running = False
    suc = True
    fail = False

    action = np.array([0.,0.])
    obj_pos = np.array([0., 0., 0.])

    S = A = T = []
    drop_counter = 0
    T0 = 0.
    start_rot = 0.


    images = []
    I = []
    pos_image = np.array([0,0])
    Simage = []
    Timages = []
    
    def __init__(self):
        rospy.init_node('rollout_recorder', anonymous=True)

        rospy.Subscriber('/hand_control/obj_pos', Float32MultiArray, self.callbackObj)

        rospy.Subscriber('/hand_control/drop', Bool, self.callbackObjectDrop)
        rospy.Subscriber('/rollout/action', Float32MultiArray, self.callbackAction)
        rospy.Subscriber('/rollout/fail', Bool, self.callbacFail)

        rospy.Service('/rollout/record_trigger', SetBool, self.callbackTrigger)
        rospy.Service('/rollout/get_states', gets, self.get_states)

        obs_srv = rospy.ServiceProxy('/observation', observation)
        self.pose_predict = rospy.ServiceProxy('/pose_nn/predict', one_predict)

        if record_images:
            # TODO: edit
            self.bridge = CvBridge()
            rospy.Subscriber('/camera/image_raw', Image, self.callbackImage, queue_size=1)

        print('[rollout_recorder] Ready ...')

        if not record_images:
            self.rate = rospy.Rate(15) # 10hz
        else:
            self.rate = rospy.Rate(10) # 2hz

        while not rospy.is_shutdown():

            if self.running:

                self.state = np.array(obs_srv().state)

                if not self.start_rot:
                    self.start_rot = self.state[2]
                    self.state[2] = self.start_rot

                self.state[2] = self.state[2] - self.start_rot

                if self.state[2] > np.pi: self.state[2] -= 2. * np.pi
                if self.state[2] < -np.pi: self.state[2] += 2. * np.pi


                self.obj_pos_nn = np.array(self.pose_predict(self.state.reshape(-1, 1)).pose)[:3]


                self.S.append(self.state)
                self.A.append(self.action)
                self.T.append(rospy.get_time() - self.T0)

                if record_images:
                    self.images.append(self.I)
                    self.Timages.append(rospy.get_time() - self.T0)

                if self.fail:
                    print('[rollout_recorder] Episode ended with %d points.'%len(self.S))
                    self.running = False

            self.rate.sleep()

    def callbackObjectDrop(self, msg):
        if (msg.data):
            self.drop_counter += 1
        else:
            self.drop_counter = 0


    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbacFail(self, msg):
        self.fail = msg.data

    def callbackImage(self, msg):
        # self.I = msg.data
        self.I = np.fromstring(msg.data, np.uint8).reshape(480,640,3)
        # self.I = cv2.imdecode(self.I, -1)
        # print self.I.shape
        # plt.imshow(self.I)
        # plt.show()

    def callbackPosImage(self, msg):
        self.pos_image = np.array([msg.position.x, msg.position.y])

    def callbackTrigger(self, msg):
        self.running = msg.data
        if self.running:
            self.S = []
            self.A = []
            self.T = []
            self.T0 = rospy.get_time()
            self.images = []
            self.Timages = []
            print('[rollout_recorder] Recording started.')
            self.fail = False
            self.start_rot = 0.0
        else:
            print('[rollout_recorder] Recording stopped with %d points.'%len(self.S))

        return {'success': True, 'message': ''}

    def get_states(self, msg):
        path = os.path.dirname(os.path.abspath(__file__))
        if record_images:
            with open(path + str(np.random.randint(10000)) + '.pkl', 'wb') as f:
                pickle.dump([self.Timages, self.Simage, self.images, self.S, self.A, self.T], f)          

        return {'states': np.array(self.S).reshape((-1,)), 'actions': np.array(self.A).reshape((-1,)), 'time': np.array(self.T).reshape((-1,))}

    def medfilter(self, x, W):
        print('[rollout_recorder] Smoothing data...')
        w = int(W/2)
        x_new = np.copy(x)
        for i in range(0, x.shape[0]):
            if i < w:
                x_new[i] = np.mean(x[:i+w])
            elif i > x.shape[0]-w:
                x_new[i] = np.mean(x[i-w:])
            else:
                x_new[i] = np.mean(x[i-w:i+w])
        return x_new

if __name__ == '__main__':
    try:
        rolloutRecorder()
    except rospy.ROSInterruptException:
        pass