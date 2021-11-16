#!/usr/bin/python 

'''
Author: Avishai Sintov
Updated by Osher Azulay
'''

import rospy
import numpy as np 
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool, Float32
from std_srvs.srv import Empty, EmptyResponse
from openhand_node.srv import MoveServos, ReadTemperature
from hand_control.srv import TargetAngles, IsDropped, observation, close, ObjOrientation, RegraspObject, TargetPos
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from common_msgs_gl.srv import SendDoubleArray, SendBool
import geometry_msgs.msg
import math
import time
import tf

class hand_control():

    finger_initial_offset = np.array([0., 0.])
    finger_opening_position = np.array([0.0, 0.0])
    finger_closing_position = np.array([0., 0.])
    finger_move_offset = np.array([0.01, 0.01])
    closed_load = np.array(20.)

    # Gripper properties
    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    gripper_flex = np.array([0., 0.])
    gripper_force = np.array([0., 0.])
    gripper_temperature = np.array([0., 0.])
    target_pos = np.array([0.,0.])
    gripper_status = 'open'

    slip_angle = 0.0
    angle = 0.0
    rot_angle = 0.0
    base_pos = [0,0]
    base_theta = 0
    calib_angle = 0
    base_height = -1.0e3
    obj_pos = [0,0]
    obj_height = -1.0e3
    obj_grasped_height = 1.0e3

    R = []
    count = 1
    max_load = 180.0
    max_drop = 5
    drop_counter = 0

    object_grasped = False



    def __init__(self):

        rospy.init_node('hand_control', anonymous=True)
        
        if rospy.has_param('~finger_initial_offset'):
            self.finger_initial_offset = rospy.get_param('~finger_initial_offset')
            self.finger_opening_position = rospy.get_param('~finger_opening_position')
            self.finger_closing_position = rospy.get_param('~finger_closing_position')
            self.finger_move_offset = rospy.get_param('~finger_move_offset')
            self.closed_load = rospy.get_param('~finger_close_load')
            self.max_load = rospy.get_param('~finger_max_load')
            self.sim_step = rospy.get_param('~simulation_step')

        # Gripper related
        rospy.Subscriber('/gripper/force', Float32MultiArray, self.callbackGripperForce)
        rospy.Subscriber('/gripper/flex', Float32MultiArray, self.callbackGripperFlex)
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/gripper/temperature', Float32MultiArray, self.callbackGripperTemp)

        rospy.Subscriber('/marker_tracker/rigid_bodies/object/pose', PoseStamped, self.callbackObjectMarkers)
        rospy.Subscriber('/marker_tracker/rigid_bodies/world/pose', PoseStamped, self.callbackWorldMarkers)


        pub_gripper_status = rospy.Publisher('/hand_control/gripper_status', String, queue_size=10)
        pub_drop = rospy.Publisher('/hand_control/drop', Bool, queue_size=10)
        pub_obj_pos = rospy.Publisher('/hand_control/obj_pos', Float32MultiArray, queue_size=10)
        pub_world_pos = rospy.Publisher('/hand_control/world_pos', Float32MultiArray, queue_size=10)
        pub_world_orientation = rospy.Publisher('/hand_control/world_orientation', Float32MultiArray, queue_size=10)
        pub_obj_orientation = rospy.Publisher('/hand_control/object_orientation', Float32, queue_size=10)
        pub_obj_height = rospy.Publisher('/hand_control/object_height', Float32, queue_size=10)

        rospy.Service('/OpenGripper', Empty, self.OpenGripper)
        rospy.Service('/CloseGripper', close, self.CloseGripper)
        rospy.Service('/RandomCloseGripper', close, self.RandomCloseGripper)

        rospy.Service('/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/IsObjDropped', IsDropped, self.CheckDroppedSrv)
        rospy.Service('/observation', observation, self.GetObservation)
        rospy.Service('/TargetPos', TargetPos, self.GetTargetPos)

        self.move_servos_srv = rospy.ServiceProxy('/openhand_node/move_servos', MoveServos)
        self.temperature_srv = rospy.ServiceProxy('/openhand_node/read_temperature', ReadTemperature)

        msg = Float32MultiArray()
        msg_ = Float32()
        self.tl = tf.TransformListener()
        self.rate = rospy.Rate(100)
        c = True
        count = 0

        while not rospy.is_shutdown():

            pub_gripper_status.publish(self.gripper_status)

            msg.data = self.obj_pos
            pub_obj_pos.publish(msg)

            msg_.data = self.obj_height
            pub_obj_height.publish(msg_)

            if count > 1000: #
                dr, verbose = self.CheckDropped()
                pub_drop.publish(dr)
                # rospy.loginfo(dr)
                # rospy.loginfo(verbose)

            count += 1

            if c and not np.all(self.gripper_load==0): # Wait till openhand services ready and set gripper open pose
                self.moveGripper(self.finger_opening_position)
                c = False

            self.rate.sleep()

    ######
    # TODO: Subscribers
    ######
    def callbackGripperPos(self, msg):
        self.gripper_pos = np.array(msg.data)

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackGripperForce(self, msg):
        self.gripper_force = np.array(msg.data)

    def callbackGripperFlex(self, msg):
        self.gripper_flex = np.array(msg.data)

    def callbackGripperTemp(self, msg):
        self.gripper_temperature = np.array(msg.data)

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
            self.base_height = msg.pose.position.z
        except:
            self.base_pos = np.array([np.nan, np.nan])
            self.base_theta = np.nan
            self.base_height = np.nan

    def callbackObjectMarkers(self, msg):
        #                                       target   source
        # The direction of the transform returned will be from the target_frame to the source_frame.
        # Which if applied to data, will transform data in the source_frame into the target_frame
        (trans, rot) = self.tl.lookupTransform("world", "object", rospy.Time(0))
        try:
            quaternion = rot
            roll,pitch,yaw = euler_from_quaternion(quaternion)
            self.angle = np.array(np.arctan2(trans[0],trans[1]))
            # self.angle -= self.calib_angle
            # if not self.calib_angle:
            #     self.calib_angle = self.angle

            self.rot_angle = yaw
            self.obj_pos = np.array([trans[1], trans[0]])
            self.obj_height = trans[2]
            self.slip_angle = math.sqrt(roll ** 2 + pitch ** 2) * 180 / math.pi
        except:
            self.obj_pos = np.array([np.nan, np.nan])
            self.angle = np.nan
            self.rot_angle = np.nan
            self.obj_height = np.nan

    ######
    # TODO: Services
    ######

    def OpenGripper(self, msg):

        self.moveGripper(self.finger_opening_position, open=True)

        self.gripper_status = 'open'

        return EmptyResponse()

    def CloseGripper(self, msg):
        # GRASP OBJECT
        # CHECK OVERHEAT
        count = 0
        if np.any(self.gripper_temperature > 52.):
            rospy.logerr('[hand_control] Actuators overheated, taking a break...')
            while 1:
                if np.all(self.gripper_temperature < 60.):
                    break
                self.rate.sleep()

        closed_load = self.closed_load

        self.object_grasped = False
        for i in range(200):
            # CHECK LOAD
            if abs(self.gripper_load[0]) > closed_load or abs(self.gripper_load[1]) > closed_load:
                count += 1
            if count > 5:
                rospy.loginfo('[hand_control] Object grasped.')
                self.gripper_status = 'closed'
                break
            # CHECK GRASP ANGLE LIMITS
            desired = self.gripper_pos + np.array([a for a in self.finger_move_offset]) # *10 / 18
            if desired[0] > 0.7 or desired[1] > 0.7:
                rospy.logerr('[hand_control] Desired angles out of bounds to grasp object.')
                break

            self.moveGripper(desired)
            rospy.sleep(0.05)

        self.rate.sleep()

        # Verify grasp based on height - not useful if camera cannot see
        print('[hand_control] Object height relative to gripper : '+ str(self.obj_height))
        if abs(self.obj_height) < 7.0e-2:
           self.object_grasped = True
           self.obj_grasped_height = self.obj_height

        print('[hand_control] Gripper actuator angles: ' + str(self.gripper_pos))
        self.rate.sleep()
        self.target_pos = self.gripper_pos

        return {'success': self.object_grasped}

    def RandomCloseGripper(self, msg):
        # GRASP OBJECT
        # CHECK OVERHEAT
        count = 0
        delayed_finger = np.random.choice(2)
        delayed_interval = np.random.choice(50)

        if np.any(self.gripper_temperature > 52.):
            rospy.logerr('[hand_control] Actuators overheated, taking a break...')
            while 1:
                if np.all(self.gripper_temperature < 60.):
                    break
                self.rate.sleep()

        closed_load = self.closed_load

        self.object_grasped = False
        for i in range(200):
            # CHECK LOAD
            if abs(self.gripper_load[0]) > closed_load or abs(self.gripper_load[1]) > closed_load:
                count += 1
            if count > 5:
                rospy.loginfo('[hand_control] Object grasped.')
                self.gripper_status = 'closed'
                break
            # CHECK GRASP ANGLE LIMITS

            desired = self.gripper_pos + np.array([a for a in self.finger_move_offset])
            if i < delayed_interval:
                desired[delayed_finger] = 0

            if desired[0] > 0.7 or desired[1] > 0.7:
                rospy.logerr('[hand_control] Desired angles out of bounds to grasp object.')
                break

            self.moveGripper(desired)
            rospy.sleep(0.05)

        self.rate.sleep()

        # Verify grasp based on height - not useful if camera cannot see
        print('[hand_control] Object height relative to gripper : '+ str(self.obj_height))
        if abs(self.obj_height) < 7.0e-2:
           self.object_grasped = True
           self.obj_grasped_height = self.obj_height

        print('[hand_control] Gripper actuator angles: ' + str(self.gripper_pos))
        self.rate.sleep()
        self.target_pos = self.gripper_pos

        return {'success': self.object_grasped}

    def MoveGripper(self, msg):
        # This function should accept a vector of normalized increments to the current angles:
        # msg.angles = [dq1, dq2], where dq1 and dq2 can be equal to 0 (no move), 1,-1 (increase or decrease angles by finger_move_offset)
        f = 2 #2 # 50.0
        inc = np.array(msg.angles)
        inc_angles = np.multiply(self.finger_move_offset, inc) # 0.01 * angle

        t = rospy.get_time()
        while rospy.get_time() - t < 0.1:
            self.target_pos += inc_angles * 1.0/f # self.gripper_pos #
            suc = self.moveGripper(self.target_pos)

        # self.target_pos += inc_angles * 1.0/f # self.gripper_pos #
        # suc = self.moveGripper(self.target_pos)
        # self.target_pos = self.gripper_pos + inc_angles * 1.0/f
        return {'success': suc}
    
    def moveGripper(self, angles, open=False):

        # TODO: add simple check allowed action function
        if not open:
            if angles[0] > 0.9 or angles[1] > 0.9 or angles[0] < -0.02 or angles[1] < -0.02:
                rospy.logerr('[hand_control] Move Failed. Desired angles out of bounds.')
                return False

            if abs(self.gripper_load[0]) > self.max_load or abs(self.gripper_load[1]) > self.max_load:
                rospy.logerr('[hand_control] Move failed. Pre-overload.')
                return False

        self.move_servos_srv.call(angles)
        # rospy.sleep(0.05)
        return True

    def all_close(self, goal, actual, tolerance):
        """
        Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        return True

    def CheckDropped(self):

        try:
            if self.gripper_pos[0] > 0.9 or self.gripper_pos[1] > 0.9 or self.gripper_pos[0] < -0.05 or self.gripper_pos[1] < -0.05:
                verbose = '[hand_control] Desired angles out of bounds -  assume dropped.'
                return True, verbose
        except:
            print('[hand_control] Error with gripper_pos.')

            # Check load
            if abs(self.gripper_load[0]) > self.max_load or abs(self.gripper_load[1]) > self.max_load:
                verbose = '[hand_control] Pre-overload.'
                return True, verbose


        if self.obj_height < -0.02:
            verbose = '[hand_control] Object is dropped. height: ' + str(self.obj_height)
            return True, verbose

        if self.obj_height > 0.2:
            verbose = '[hand_control] Object is too high. height: ' + str(self.obj_height)
            return True, verbose

        if self.slip_angle > 25:
            verbose = '[hand_control] Object slipped. slip_angle: ' + str(self.slip_angle)
            return True, verbose

        return False, ''

    def CheckDroppedSrv(self, msg):

        dr, verbose = self.CheckDropped()
        
        if len(verbose) > 0:
            rospy.logerr(verbose)

        return {'dropped': dr}

    def GetObservation(self, msg):
        obs = np.concatenate((self.obj_pos, # 2                        0,1
                              np.expand_dims(self.rot_angle, 0), #     2
                              self.gripper_pos, # 2                    3,4
                              self.gripper_load, # 2                   5,6
                              self.gripper_force ))  # 2                 7,8
        return {'state': obs}

    def GetTargetPos(self, msg):
        return {'angle': self.target_pos}


if __name__ == '__main__':
    
    try:
        hand_control()
    except rospy.ROSInterruptException:
        pass
