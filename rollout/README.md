 <!--:construction_worker: :construction: **_This page is under construction_** :construction: :construction_worker:-->


# Package for running rollouts with an underactuated robotic hand

This page provides instructions and source code for working with underactuated adaptive hands. This is complementary material to the paper

> ***Learning Haptic-based Object Pose Estimation\\ for In-hand Manipulation with Underactuated Robotic Hands***

submitted to the *IEEE Robotics and Automation Letters (RA-L) journal with the ICRA 2022*.

**The code is based on ROS and tested on Melodic-devel.**

To build the hands, follow instructions in the [Yale OpenHand project](https://www.eng.yale.edu/grablab/openhand/).

To build the flx-finger, follow instructions in the [flx-finger](https://github.com/osheraz/flx-tac-finger).

To launch the hand control, please go to the ```hand_control``` package.

---
## Rollout node

In order to launch a rollout collection automatically:
- Run:
     ```
     roslaunch rollout rollout.launch
     rosrun rollout run_rollout.py
     ```
  
#### Available dataset:

To generate the dataset, we used several simple action sequences to explore the workspace of the hand.
During execution of the plans, we randomly applied random actions sequences to further explore the hand workspace through frequent changes of action.

The dataset we used in our study can be accessed through ```data\rollout_plans```.\
We recorded six prismatic objects with different cross-sections:
- `cyl30`: 30mm diameter circle
- `cyl20`: 20mm diameter circle
- `sqr`: 20x20mm square
- `ellipse`: Elliptical object with r1= 40mm, r2= 20mm
- `crescent`: Crescent-shaped object with r_out=50mm, r_in=40mm
- `obj`: Arbitrary shaped object

Example for accessing a transition data files:
```python
import numpy as np
import pickle

with open('states_01-09-2021_19-16-31_itr_3.pickle', 'rb') as f: 
    episode = pickle.load(f)
    obs, acs, next_obs, time = episode['observation'], episode['action'], episode['next_observation'],  episode['time']

```

An observation is formed in the following format:
```
[<object position {x,y} (2)>, <object orientation (1)>, <actuaturs angles   (2), actuaturs loads (2)> , flx-reading (2) ]
```
which we later divides into input\output features.