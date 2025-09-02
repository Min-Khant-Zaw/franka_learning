from typing import Dict, List

import torch
import numpy as np
import pickle

import torchcontrol as toco
from torchcontrol.utils import to_tensor, stack_trajectory

from polymetis import RobotInterface
from polymetis.utils.data_dir import get_full_path_to_urdf

import hydra
import time
import sys, select

from utils.environment import Environment
from utils.experiment_utils import ExperimentUtils
from utils.trajectory import Trajectory
from utils.my_pybullet_utils import *

from tf.transformations import euler_from_quaternion

class JointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space.
    """
    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # Parse current state
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        with torch.no_grad():
            self.joint_pos_desired.copy_(joint_pos_current.view_as(self.joint_pos_desired))
            self.joint_vel_desired.zero_()

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}
        

@hydra.main(config_path="../config", config_name="demo_recorder")
def main(cfg):

    # Initialize robot interface
    robot = RobotInterface(
        ip_address = cfg.ip
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
        robot_model=robot_model,
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

    policy = JointImpedanceControl(
        joint_pos_current=robot.get_joint_positions(),
        Kp=default_kq,
        Kd=default_kqd,
        robot_model=robot_model
    )

    state_log = robot.send_torch_policy(policy, blocking=False)

    print("Move the robot by hand, press ENTER to stop.")
    
    while robot.is_running_policy():
        # Check if the user pressed ENTER
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = input()
            policy.set_terminated()
            break

        # Update the experiment utils executed trajectory tracker
        curr_pos = robot.get_joint_positions().numpy()
        timestamp = time.time() - start_time
        expUtil.update_tracked_traj(timestamp, curr_pos)
    
    print("----------------------------------")

    # Process and save the recording
    raw_demo = expUtil.tracked_traj[:,1:8]

    print(f"Raw demo: {raw_demo}")
    print(f"Size of raw demo: {len(raw_demo)}\n")
            
    # Trim ends of waypoints where the human didn’t move the robot significantly and create Trajectory
    time_window = 0.3
    n_samples = max(1, int(round(time_window * hz)))
    N = raw_demo.shape[0]
    lo = 0
    hi = N - 1
    # print(f"High index: {hi}")
    while (lo + n_samples < N) and (np.linalg.norm(raw_demo[lo + n_samples] - raw_demo[lo]) < 0.01):
        print(f"[DEBUG] Norm between two positions for low index: {np.linalg.norm(raw_demo[lo + n_samples] - raw_demo[lo])}")
        lo += 1
        # print(f"Low: {lo}")
    while (hi - n_samples >= 0) and (np.linalg.norm(raw_demo[hi] - raw_demo[hi - n_samples]) < 0.01):
        print(f"[DEBUG] Norm between two positions for high index: {np.linalg.norm(raw_demo[hi] - raw_demo[hi - n_samples])}")
        hi -= 1
        # print(f"High: {hi}")
    print(f"Low: {lo}")
    if lo >= hi:
        waypts = raw_demo[lo:lo+2, :]
    else:
        waypts = raw_demo[lo:hi+1, :]
    print(f"Waypoints: {waypts}")
    waypts_time = np.linspace(0.0, T, waypts.shape[0])
    traj = Trajectory(waypts, waypts_time)

    # Downsample/Upsample trajectory to fit desired timestep and T.
    num_waypts = int(T / timestep) + 1
    if num_waypts < len(traj.waypts):
        demo = traj.downsample(int(T / timestep) + 1)
    else:
        demo = traj.upsample(int(T / timestep) + 1)

    # Downsample trajectory for visualization
    viz_traj = demo.downsample(100)
    viz_traj_waypts = viz_traj.waypts
    
    # Visualize Trajectory
    visualizeTraj(environment, viz_traj_waypts, radius=0.02, color=[0, 0, 1, 1])

    # Decide whether to save trajectory
    line = input("Type [yes/y/Y] if you're happy with the demonstration: ")
    line = line.lower()
    if (line != "yes") and (line != "y"):
        print("Not happy with demonstration. Terminating experiment.\n")
    else:
        ID = input("Please type in the ID number (e.g. [0/1/2/...]): ")
        task = input("Please type in the task number (e.g. [0/1/2/...]): ")
        filename = "demo" + "_ID" + ID + "_task" + task
        savefile = expUtil.get_unique_filepath("demo", filename)
        pickle.dump(demo, open(savefile, "wb"))
        print(f"Saved demonstration in {savefile}.")

if __name__ == "__main__":
    main()