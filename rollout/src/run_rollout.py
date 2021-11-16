#!/usr/bin/env python

# Written by Avishai Sintov
# Modified by Osher Azulay

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout.srv import rolloutReq, rolloutReqMod
import time
import glob
from scipy.io import savemat, loadmat
import os
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool, Float32

gripper_force = np.array([0., 0.])

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReqMod)

rospy.init_node('run_rollout_set', anonymous=True)
rate = rospy.Rate(15) # 15hz
state_dim = 9
action_dim = 2

path = 'plans/'
rollout = True
save_graph = True
save_trans = True




def medfilter(x, W):
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

def clean(D):
    print('Cleaning data...')

    i = 0
    inx = []
    while i < D.shape[0]:
        # for j in range(D.shape[1])
        # if i > 5 and np.linalg.norm( D[i, 6:8] - D[i-5, 6:8] ) > 4:
        #     i += 1
        #     continue
        if i > 0 and np.linalg.norm(D[i,3] - D[i-1,3])*180/3.14 > 5.0:
            i += 1
            continue
        inx.append(i)
        i += 1

    return D[inx,:]

def save_display(d,tp, tofile):

    # Medfilter or clean
    # d = clean(d)
    # for j in range(d.shape[1]):
    #     d[:, j] = medfilter(d[:, j], 20)
    # if len(d) < 100:
    #     continue

    end = -10
    d = d[:end, :]
    x = 0
    y = 1

    fig = plt.figure(figsize=(10,10))

    fig.add_subplot(3, 2, 1)
    plt.plot(d[0, x], d[0, y], 'r*', label='start')

    # plt.plot(d[:,x],d[:,y],'.')
    plt.plot(d[-1, x], d[-1, y], 'k*', label='end')

    # Arrow
    for s in range(len(d) - 3):
        if s % 2:
            plt.arrow(d[s, x],
                      d[s, y],
                      d[s + 3, x] - d[s, x],
                      d[s + 3, y] - d[s, y],
                      head_width=0.001,
                      width=0.0,
                      ec='green',
                      facecolor='green',
                      alpha=0.5)

    # Circle
    # for j in range(0,len(d),len(d)/10):
    #     r = 0.02/2
    #     circle = plt.Circle((d[j,x],d[j,y]), r, color='grey', alpha=(j+0.05)/float(len(d)))
    #     ax = plt.gca()
    #     ax.add_patch(circle)
    #     x_values = [d[j,x], d[j,x]+r*np.cos(d[j, 2])]
    #
    #     y_values = [d[j,y], d[j,y]+r*np.sin(d[j, 2])]
    #     plt.plot(x_values, y_values,'k')

    plt.title('  num points:' + str(len(d)))
    # plt.legend()
    plt.axis([-0.08, 0.08, 0.07, 0.17], 'equal')

    plt.grid()
    plt.gca().set_aspect("equal")

    fig.add_subplot(3, 2, 2)

    plt.title('YAW')
    plt.plot(d[:, 2] * 180 / 3.14, 'o', markersize=1)

    fig.add_subplot(3, 2, 3)
    #         obs = np.concatenate((self.obj_pos,                               2   0,1
    #                               self.rot_angle,                             1   2
    #                               self.gripper_pos,                           2   3,4
    #                               self.gripper_load,                          2   5,6
    #                               self.gripper_force))                        2   7,8

    plt.title('Motor load')
    plt.plot(d[:, 5], 'o', markersize=1)
    plt.plot(d[:, 6], 'o', markersize=1)

    fig.add_subplot(3, 2, 4)

    plt.title('Force')
    plt.plot(d[:, 7], 'o', markersize=1)
    plt.plot(d[:, 8], 'o', markersize=1)

    fig.add_subplot(3, 2, 5)
    plt.title('motor angle')
    plt.plot(d[:, 3], 'o', markersize=1)
    plt.plot(d[:, 4], 'o', markersize=1)

    fig.add_subplot(3, 2, 6)
    plt.title('target pos')
    plt.plot(tp[:, 0], 'o', markersize=1)
    plt.plot(tp[:, 1], 'o', markersize=1)
    fig.savefig(tofile)


if rollout:

    params = np.array([rospy.get_param('/openhandNode/motor_offset')[0],
                       rospy.get_param('/openhandNode/motor_offset')[1],
                       rospy.get_param('/hand_control/finger_close_load')])

    files = glob.glob(path + "*.txt")
    logdir_prefix = 'rollout'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    for i in range(len(files)):

        action_file = files[i]
        npfile = action_file[:-4]
        logdir = logdir_prefix + '_' + npfile
        index = time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\nLOGGING TO: " +logdir +  "\n")

        print('Rolling-out file number ' + str(i+1) + ': ' + action_file + '.\n')

        A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

        matches = ["4", "8", "2"]

        if any(x in action_file for x in matches):
            Af = A.reshape((-1,))
            Af *= 10
        else:
            A[:10, :] *= 0.5
            Af = A.reshape((-1,))
            Af /= 10

        for j in range(8):
            # conduct j numbers of each rollout
            rospy.sleep(2)

            msg = rospy.wait_for_message('/gripper/force', Float32MultiArray)
            gripper_force = msg.data

            print('Intial finger load:\t' + str(gripper_force))

            resp = rollout_srv(Af)

            S = np.array(resp.states).reshape(-1, state_dim)
            NS = np.array(resp.next_states).reshape(-1, state_dim)
            A = np.array(resp.actions).reshape(-1, action_dim)
            T = np.array(resp.time).reshape(-1, 1)
            TP = np.array(resp.target_pos).reshape(-1, action_dim)

            rospy.sleep(2)

            S[:, 7] -= gripper_force[0]
            NS[:, 7] -= gripper_force[0]

            S[:, 8] -= gripper_force[1]
            NS[:, 8] -= gripper_force[1]

            S = np.c_[S, params * np.ones((len(S), len(params)))]  # TODO:  Adding the initial params
            NS = np.c_[NS, params * np.ones((len(NS), len(params)))]

            transition = {"observation": S[:, :],
                         "action": A[:, :],
                         "target_pos": TP[:, :],
                         "next_observation": NS[:, :],
                         "time": T[:],
                         "success": resp.success,
                         "reason": resp.reason}

            print('suc: ' + resp.reason + '\tnum_actions: '+ str(len(A)))

            if len(S) <= 10:
                continue

            file_name = 'states_' + index + '_itr_' + str(j)
            np.save(logdir + '/' + file_name + '.npy', S)

            if save_graph:
                graphs_dir = logdir + '/graphs'
                if not (os.path.exists(graphs_dir)):
                    os.makedirs(graphs_dir)
                save_display(S,TP, graphs_dir + '/' + file_name)

            if save_trans:
                trans = logdir + '/transition'
                if not (os.path.exists(trans)):
                    os.makedirs(trans)
                with open(trans + '/' + file_name + '.pickle', 'wb') as handle:
                    pickle.dump(transition, handle, protocol=pickle.HIGHEST_PROTOCOL)


