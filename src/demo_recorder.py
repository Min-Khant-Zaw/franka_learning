from typing import Dict, List

import torch
import numpy as np
import pickle

import torchcontrol as toco
from torchcontrol.utils import to_tensor, stack_trajectory
from torchcontrol.policies import HybridJointImpedanceControl

from polymetis import RobotInterface
from polymetis.utils.data_dir import get_full_path_to_urdf

import hydra
import time
import sys, select

from planners.trajopt_planner import TrajOpt
from utils.environment import Environment
from utils.experiment_utils import ExperimentUtils
from utils.trajectory import Trajectory
from utils.my_pybullet_utils import *

from tf.transformations import euler_from_quaternion

@hydra.main(config_path="../config", config_name="demo_recorder")
def main(cfg):

    # Initialize robot interface
    robot = RobotInterface(
        ip_address = "localhost"
    )

    # Get robot metadata
    default_kq = torch.Tensor(robot.metadata.default_Kq)
    default_kqd = torch.Tensor(robot.metadata.default_Kqd)
    default_kx = torch.Tensor(robot.metadata.default_Kx)
    default_kxd = torch.Tensor(robot.metadata.default_Kxd)
    hz = robot.metadata.hz

    # ----- General Setup ----- #
    start = np.array(cfg.setup.start)
    object_centers = cfg.setup.object_centers
    T = cfg.setup.T
    num_steps = int(T * hz)
    timestep = T / num_steps
    save_dir = cfg.setup.save_dir

    # Utilities for recording data.
    expUtil = ExperimentUtils(save_dir)

    # Get robot model configuration
    robot_model_cfg = cfg.robot_model

    # Get the urdf file of Franka Panda
    robot_description_path = get_full_path_to_urdf(
            robot_model_cfg.robot_description_path
        )

    # Get robot model from torchcontrol.models
    robot_model = toco.models.RobotModelPinocchio(
        urdf_filename=robot_description_path,
        ee_link_name=robot_model_cfg.ee_link_name
    )

    # Create Pybullet environment
    environment = Environment(
        robot_model_cfg=robot_model_cfg,
        object_centers=object_centers,
        gui=True
    )

    # Reset
    robot.go_home(time_to_go=T)

    # Move the robot to start
    start = to_tensor(start)
    print(f"\nMoving joints to start: {start} ...\n")
    state_log = robot.move_to_joint_positions(positions=start, time_to_go=T)

    start_time = time.time()
    print(f"\nStart time: {start_time}")

    policy = HybridJointImpedanceControl(
        joint_pos_current=robot.get_joint_positions(),
        Kq=default_kq,
        Kqd=default_kqd,
        Kx=default_kx,
        Kxd=default_kxd,
        robot_model=robot_model
    )

    state_log = robot.send_torch_policy(policy, blocking=False)

    print("Move the robot by hand, press ENTER to stop.")
    
    while robot.is_running_policy():
        # Check if the user pressed ENTER
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = input()
            policy.set_terminated()

        # Update the experiment utils executed trajectory tracker
        curr_pos = robot.get_joint_positions().numpy()
        timestamp = time.time() - start_time
        expUtil.update_tracked_traj(timestamp, curr_pos)
    
    print("----------------------------------")

    # Process and save the recording
    raw_demo = expUtil.tracked_traj[:,1:8]
            
    # Trim ends of waypoints where the human didn’t move the robot significantly and create Trajectory
    lo = 0
    hi = raw_demo.shape[0] - 1
    while np.linalg.norm(raw_demo[lo] - raw_demo[lo + 1]) < 0.01 and lo < hi:
        lo += 1
    while np.linalg.norm(raw_demo[hi] - raw_demo[hi - 1]) < 0.01 and hi > 0:
        hi -= 1
    waypts = raw_demo[lo:hi+1, :]
    waypts_time = np.linspace(0.0, T, waypts.shape[0])
    traj = Trajectory(waypts, waypts_time)

    # Downsample/Upsample trajectory to fit desired timestep and T.
    num_waypts = int(T / timestep) + 1
    if num_waypts < len(traj.waypts):
        demo = traj.downsample(int(T / timestep) + 1)
    else:
        demo = traj.upsample(int(T / timestep) + 1)

    # Decide whether to save trajectory
    visualizeTraj(environment, demo.waypts, radius=0.05, color=[0, 0, 1, 1])
    line = input("Type [yes/y/Y] if you're happy with the demonstration: ")
    line = line.lower()
    if (line != "yes") and (line != "y"):
        print("Not happy with demonstration. Terminating experiment.\n")
    else:
        ID = input("Please type in the ID number (e.g. [0/1/2/...]): ")
        task = input("Please type in the task number (e.g. [0/1/2/...]): ")
        filename = "demo" + "_ID" + ID + "_task" + task
        savefile = expUtil.get_unique_filepath("demo", filename)
        pickle.dump(demo, open(savefile, "wb" ))
        print("Saved demonstration in {}.".format(savefile))

if __name__ == "__main__":
    main()