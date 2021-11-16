#!/usr/bin/env python
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import random
import glob

from scipy.signal import savgol_filter

path = "/home/osher/catkin_ws/src/manipulations/rollout/data/rollout_plans/obj/plan"

print('[data_load] Loading data from "' + path[:-5] + '"...')
total_files = []
first = 1
amount = 0

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

#         obs = np.concatenate((self.obj_pos,                               2   0,1
#                               np.array([self.angle, self.rot_angle]),     2   2,3
#                               self.gripper_pos,                           2   4,5
#                               self.gripper_load,                          2   6,7
#                               self.gripper_flex,                          2   8,9
#                               self.gripper_force))                        2   10,11

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


skip = 0

if True:
    for i in range(1,11):
        files = glob.glob(path + str(i) + "/*.npy")
        for i in range(int(len(files)/1)):
            if files[i].find('total') < 0:
                total_files.append(files[i])
                if first:
                    data = np.load(files[i], allow_pickle=True, encoding='latin1')
                    # data = np.concatenate(data[:])
                    first = 0
                d = np.load(files[i], allow_pickle=True, encoding='latin1')

                # d = clean(d)
                # for j in range(d.shape[1]):
                #     d[:,j] = medfilter(d[:,j], 20)


                if len(d) < 21:
                    continue

                # if np.any(d[:, 1] <= 0.085):
                #     continue

                end = -20
                d = d[:end,:]
                x = 0
                y = 1
                amount += d.shape[0]
                plt.subplot(2,1,1)
                # plt.plot(d[0,x],d[0,y],'r*',label='start')
                # #
                # # # plt.plot(d[:,x],d[:,y],'.')
                # plt.plot(d[-1,x],d[-1,y],'k*',label='end')
                # #
                # # #Arrow
                # for s in range(len(d)-3):
                #     if s%2:
                #         plt.arrow(d[s,x],
                #                   d[s,y],
                #                   d[s+3,x] - d[s,x],
                #                   d[s+3,y] - d[s,y],
                #                   head_width=0.0001,
                #                   width=0.0,
                #                   ec ='green',
                #                   facecolor='green',
                #                   alpha=0.5)

                #Circle
                # for j in range(0,len(d),len(d)/10):
                #     r = 0.02/2
                #     circle = plt.Circle((d[j,x],d[j,y]), r, color='grey', alpha=(j+0.05)/float(len(d)))
                #     ax = plt.gca()
                #     ax.add_patch(circle)
                #     x_values = [d[j,x], d[j,x]+r*np.cos(d[j, 3])]
                #
                #     y_values = [d[j,y], d[j,y]+r*np.sin(d[j, 3])]
                #     plt.plot(x_values, y_values,'k')

                title = files[i]
                plt.title(title[-42:] + '  num points:' + str(len(d)))
                # plt.legend()
                # plt.axis([-0.08, 0.08, 0.05, 0.14], 'equal')

                plt.grid()
                plt.gca().set_aspect("equal")

                plt.subplot(2,1,2)



                start_rot = 0 * d[0,2]
                start_rot_obj = 0 * start_rot % (np.pi / 1.)  # 4 for sqr, without for cyl

                d[:, 2] = d[:, 2] - start_rot + start_rot_obj
                if True:
                    idx = []
                    sign = []
                    for k in range(2, d.shape[0]):
                        if (d[k, 2] - d[k-1, 2]) > np.pi / 2:
                            idx.append(k)
                            sign.append(-1)
                        elif (d[k, 2] - d[k-1, 2]) < -np.pi / 2:
                            idx.append(k)
                            sign.append(1)

                    if idx:
                        skip +=1
                        continue

                        if len(idx) == 1:
                            d[idx[-1]:, 2] = d[idx[-1]:, 2] + sign[-1] * 2* np.pi
                        else:
                            for ix in range(0,len(idx)-1,2):
                                d[idx[ix]:idx[ix+1], 2] = d[idx[ix]:idx[ix+1], 2] + sign[ix] * 2 *np.pi

                # for k in range(file.shape[0]):
                #     if file[k,2] > np.pi:
                #         file[k,2] -= 2. * np.pi
                #         print('1')
                #     if file[k,2] < -np.pi:
                #         file[k,2] += 2. * np.pi
                #         print('2')


                d[:,2] *= 180 / np.pi

                plt.plot(d[:, 2])
                # plt.subplot(2, 1, 1)
                # plt.plot(d[:, 8],'o', markersize=0.1)
                # plt.title('YAW')
                # plt.plot(d[:,3]*180/3.14,'o')

                # plt.subplot(2,1,2)
                #         obs = np.concatenate((self.obj_pos,                               2   0,1
                #                               np.array([self.angle, self.rot_angle]),     2   2
                #                               self.gripper_pos,                           2   3,4
                #                               self.gripper_load,                          2   5,6
                #                               self.force,                                 2   7,8

                # plt.title('Motor load')
                # plt.plot(d[:, 7],'o', markersize=0.5)
                # plt.plot(d[:, 8],'o', markersize=0.5)


                # mng = plt.get_current_fig_manager()
                # mng.resize(*mng.window.maxsize())
    print(amount)
    print(skip)
plt.show()

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import matplotlib as mpl


def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta) + xy[0],
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=False, **kwargs)

    return poly
    # obj = RotatingRectangle((-self.object_true_traj[-1, 0], self.object_true_traj[-1, 1]), width=20, height=20,
    #                         rel_point_of_rot=([10,10]),
    #                         angle=self.object_true_traj[-1, 2], color=gt_color, alpha=0.5)

    # obj2 = Arc((x, y), 2 * r, 4 * r,
    #            180 + q, 0.0, 180, color='red', alpha=0.5)


# obj = mpl.patches.Ellipse(xy=(-self.object_true_traj[-1, 0],self.object_true_traj[-1, 1]),
#                           width=2*r, height=2*r, angle= 180 + self.object_true_traj[-1, 2],  fill=True, color=gt_color, alpha = 0.5)


class RotatingArc():
    def __init__(self, xy, r1,r2, rel_point_of_rot,angle, color, alpha):
        self.rect = Arc(xy=(xy), width=r1,height=r2, color=color, alpha=alpha, angle = 180 + angle, theta1=0, theta2=180)
        # Rectangle.__init__(xy, width, height, **kwargs)
        self.rel_point_of_rot = rel_point_of_rot
        self.xy_center = self.rect.center
        self.angle = angle
        self.set_angle(self.angle)

    def _apply_rotation(self):
        angle_rad = self.angle * np.pi / 180
        m_trans = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]])

        shift = np.matmul(-m_trans, self.rel_point_of_rot)
        self.rect.center = self.xy_center + shift

    def set_angle(self, angle):
        self.angle = angle
        self._apply_rotation()

    def set_rel_point_of_rot(self, rel_point_of_rot):
        self.rel_point_of_rot = rel_point_of_rot
        self._apply_rotation()

    def set_xy_center(self, xy):
        self.xy_center = xy
        self._apply_rotation()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def middle_element(lst):
  if len(lst) % 2 != 0:
    return lst.index(range(lst) // 2)
  else:
    return len(lst) // 2 + len(lst) // 2 - 1

if False:
    fig, ax = plt.subplots()
    q = 50
    x = 50
    y = 50
    r = 10


    # obj1 = mpl.patches.Ellipse(xy=(x,y), width=2*r, height=2*r, angle= 180 + q,  fill=True, color='blue', alpha = 0.5)
    # obj2 = mpl.patches.Ellipse(xy=(x-r*np.cos(q),y-r*np.sin(q)), width=2*r, height=2*r, angle= 180 + q,  fill=True, color='white', alpha = 1)

    obj = Arc((x , y ), 2 * r, 2 * r,
               180 + q, 0.0, 180, color='red', alpha=0.5)
    obj2 = Arc((x, y), 2 * r, 4 * r,
               180 + q, 0.0, 180, color='red', alpha=0.5)

    # obj = arc_patch(xy=(x,y), width=3*r,
    #             height=3*r, theta1=q, theta2=q+180.)

    # obj2 = RotatingArc((x, y), r1=2*r, r2=2*r,
    #                         rel_point_of_rot=([0 , 0]),
    #                         angle=q, color='red', alpha=0.5)
    #
    #
    m_trans = np.array([[np.cos(np.radians(q)), -np.sin(np.radians(q))],
                        [np.sin(np.radians(q)), np.cos(np.radians(q))]])

    shift = np.matmul(m_trans, [0,0])
    # print(shift)
    x1 = x + obj.get_path().vertices[-10,0]
    y1 = y + r*np.sin(np.degrees(q))
    # print(x1,y1)
    # obj3 = RotatingArc((x, y), r1=2*r, r2=r,
    #                         rel_point_of_rot=([0 , 0]),
    #                         angle=q, color='k', alpha=0.5)

    # xy = obj.xy[obj.xy.shape[0]/2]

    ax.plot(x,y,'ro')
    ax.plot(x1,y1,'ko')

    # ax.plot(a,b,'ko')

    ax.add_artist(obj)
    ax.add_artist(obj2)
    # ax.add_artist(obj3.rect)
    # ax.add_artist(obj2.rect)

# plt.axis('equal')
# ax.set_xlim(0, 100)
# ax.set_ylim(0, 100)
