 <!--:construction_worker: :construction: **_This page is under construction_** :construction: :construction_worker:-->


# Package for underactuated hand control

This page provides instructions and source code for working with underactuated adaptive hands. This is complementary material to the paper

> ***Learning Haptic-based Object Pose Estimation for in-hand Manipulation with Underactuated Robotic Hands***


**The code is based on ROS and tested on Melodic-devel.**

To build the hands, follow instructions in the [Yale OpenHand project](https://www.eng.yale.edu/grablab/openhand/).

To build the flx-finger, follow instructions in the [flx-finger](https://github.com/osheraz/flx-tac-finger).

Before launching the hand, load the `force.ino` code into the arduino.

### Model-O with flx-finger

We use a dedicated control package to control the hand which we built on top of the  [Yale openhand-node package](https://github.com/grablab/openhand_node).

In order to control the hand:
- Setup hand parameters in `control_params.yaml`
- Run:
     ```
     roslaunch hand_control run.launch
     ```

#### Available services:
```python
rospy.Service('/OpenGripper', Empty, self.OpenGripper) # Move the gripper to the initial state
rospy.Service('/CloseGripper', close, self.CloseGripper) # Close the gripper up to pre-defined pressure
rospy.Service('/RandomCloseGripper', close, self.RandomCloseGripper) # Random-delayed-Close of the gripper up to pre-defined pressure
rospy.Service('/MoveGripper', TargetAngles, self.MoveGripper) # Apply current action to each finger of the gripper
rospy.Service('/IsObjDropped', IsDropped, self.CheckDroppedSrv) # Check if the grasped object is dropped
rospy.Service('/observation', observation, self.GetObservation) # Get the current hand feature configuration
```

The observation service return the following features:
```python
def GetObservation(self, msg):
    obs = np.concatenate((self.obj_pos,                      #  x-y pose of the grapsed object
                          np.expand_dims(self.rot_angle, 0), #  orientation of the grapsed object
                          self.gripper_pos,                  #  actuator angles
                          self.gripper_load,                 #  actuator loads
                          self.gripper_force ))              #  flx-readings
    return {'state': obs}
```python
