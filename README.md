# franka_learning: Franka HRI

Control, planning, and learning system for human-robot interaction with a Franka Emika 7DOF robotic arm. Supports learning from physical corrections and demonstrations. Reimplementation of [jaco_learning](https://github.com/andreea7b/jaco_learning) for Franka Emika robotic arm.

## Dependencies
* Ubuntu 20.04, PyBullet
* Polymetis

## Running the Adaptive Physical HRI System
### Running the server and robot client for Franka
It is always a good practice to test the code on the simulation first before running it on the hardware. To launch the server and simulation client:
```
cd src
python3 launch_robot.py robot_client=franka_sim
```

To launch the server and robot client on hardware:
```
python3 launch_robot.py robot_client=franka_hardware
```

To launch the robot client in read-only mode for debugging purposes:
```
python3 launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.readonly=true
```

### Running controllers
To run controllers in simulation, put localhost for ip. For example, for path_follower:
```
python3 path_follower.py ip="localhost"
```

**To run the controllers on hardware, it is recommended that it is run on a different machine than the one where launch_robot.py is run. Both machines should be put on the same network.**

To run controllers on hardware, put the ip address of the machine where launch_robot.py is run. For example, for path_follower:
```
python3 path_follower.py ip="172.16.0.1"
```

The script reads the corresponding Hydra configuration file `../config/path_follower.yaml` containing all important parameters. Given a start, a goal, and other task specifications, a planner plans an optimal path, then the controller executes it. 

Some important parameters for specifying the task in the yaml include:
* `start`: Franka start configuration
* `goal`: Franka goal configuration
* `goal_pose`: Franka goal pose (optional)
* `T`: Time duration of the path
* `feat_list`: List of features the robot's internal representation contains. Options: "table" (distance to table), "coffee" (coffee cup orientation), "human" (distance to human), "laptop" (distance to laptop).
* `feat_weights`: Initial feature weights.

### Learning from physical human corrections
To demonstrate planning and control with online learning from physical human corrections, run:
```
python3 phri_inference.py ip="172.16.0.1"
```

The script reads the corresponding Hydra configuration file `../config/phri_inference.yaml` containing all important parameters. Given a start, a goal, and other task specifications, a planner plans an optimal path, and the controller executes it. A human can apply a physical correction to change the way the robot is executing the task. Depending on the learning method used, the robot learns from the human torque accordingly and updates its trajectory in real-time.

### Learning from physical human demonstrations
To demonstrate learning from human demonstrations, first record some demonstrations:
```
python3 demo_recorder.py ip="172.16.0.1"
```

The script reads the corresponding Hydra configuration file `../config/demo_recorder.yaml` containing all important parameters. Given a start, the Jaco is controlled to the initial location, after which it waits for human input. Once the start is reached, the person can physically direct the arm to demonstrate how it should perform the task. The user is then shown the collected the demonstration and prompted to save it.

To perform inference from human demonstrations, run:
```
python3 demo_inference.py
```

The script reads the corresponding Hydra configuration file `../config/demo_inference.yaml`, then performs inference. There is an option to perform inference from either recorded demonstrations or a simulated one, created using the planner.

### References
* https://github.com/andreea7b/jaco_learning
* Polymetis: https://facebookresearch.github.io/fairo/polymetis/index.html