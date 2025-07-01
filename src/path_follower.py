from typing import Dict, List

import torch
import numpy as np

import torchcontrol as toco
from torchcontrol.utils.tensor_utils import to_tensor, stack_trajectory

from polymetis import RobotInterface
from polymetis.utils.data_dir import get_full_path_to_urdf

import hydra
import time

from planners.trajopt_planner import TrajOpt
from utils.environment import Environment

class PathFollower(toco.PolicyModule):
    def __init__(
            self,
            joint_pos_trajectory: List[torch.Tensor],
            joint_vel_trajectory: List[torch.Tensor],
            Kq,
            Kqd,
            Kx,
            Kxd,
            robot_model: torch.nn.Module,
            timestep,
            ignore_gravity=True
    ):
        """
        Executes a joint trajectory by using a joint PD controller to stabilize around waypoints in the trajectory.

        Args:
            joint_pos_trajectory: Desired joint position trajectory as list of tensors
            joint_vel_trajectory: Desired joint velocity trajectory as list of tensors
            Kq: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kqd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kx: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kxd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise

        (Note: nA is the action dimension and N is the number of degrees of freedom)
        """
        super().__init__()

        self.joint_pos_trajectory = to_tensor(stack_trajectory(joint_pos_trajectory))
        self.joint_vel_trajectory = to_tensor(stack_trajectory(joint_vel_trajectory))

        # Get the number of waypoints
        self.N = self.joint_pos_trajectory.shape[0]
        assert self.joint_pos_trajectory.shape == self.joint_vel_trajectory.shape

        # Control modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            robot_model=robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

        self.start_time = time.time()
        self.timestep = timestep

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Parse current state
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Query plan for desired state
        joint_pos_desired = self.joint_pos_trajectory[self.i, :]
        joint_vel_desired = self.joint_vel_trajectory[self.i, :]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            joint_pos_desired,
            joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current)
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )   # coriolis and gravity compensation
        torque_output = torque_feedback + torque_feedforward

        # Increment and terminate if all waypoints have been executed
        self.i += 1
        if self.i == self.N:
            self.set_terminated()
        # elasped_time = time.time() - self.start_time
        # self.i = min(int(elasped_time / self.timestep), self.N - 1)
        # if elasped_time == self.N * self.timestep:
        #     self.set_terminated()

        return {"joint_torques": torque_output}

def compute_joint_velocities(joint_pos_trajectory: np.ndarray, timestep: float) -> np.ndarray:
    """
    Compute the velocities of joints using finite difference of joint position trajectory

    Args:
        joint_pos_trajectory: (m, n) numpy array of desired joint position trajectory
        timestep: time between waypoints
    
    Returns:
        joint_vel_trajectory: (m, n) numpy array of desired joint velocity trajectory
    """
    m, n = np.shape(joint_pos_trajectory)
    joint_vel_trajectory = np.zeros((m, n))

    # Forward difference for the first point
    joint_vel_trajectory[0] = (joint_pos_trajectory[1] - joint_pos_trajectory[0]) / timestep

    # Backward difference for the last point
    joint_vel_trajectory[-1] = (joint_pos_trajectory[-1] - joint_pos_trajectory[-2]) / timestep

    # Central difference for the rest
    joint_vel_trajectory[1:-1] = (joint_pos_trajectory[2:] - joint_pos_trajectory[:-2]) / (2 * timestep)

    return joint_vel_trajectory

@hydra.main(config_path="../config", config_name="launch_robot")
def main(cfg):

    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost"
    )

    # Reset
    robot.go_home()

    # Create policy instance
    default_kq = torch.Tensor(robot.metadata.default_Kq)
    default_kqd = torch.Tensor(robot.metadata.default_Kqd)
    default_kx = torch.Tensor(robot.metadata.default_Kx)
    default_kxd = torch.Tensor(robot.metadata.default_Kxd)

    # Create trajectory planner
    n_waypoints = 5
    start = np.array([104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 225.0])
    goal = np.array([210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0])
    goal_pose = np.array([-0.46513, 0.29041, 0.69497])
    feat_list = ["efficiency", "table", "coffee"]
    feat_weights = [1.0, 0.0, 1.0]
    object_centers = {"HUMAN_CENTER": [0.5, -0.55, 0.9], "LAPTOP_CENTER": [-0.7929, -0.1, 0.0]}
    max_iter = 50
    T = 20.0
    timestep = 0.5

    # print(robot.metadata)

    robot_model_cfg = cfg.robot_model

    robot_description_path = get_full_path_to_urdf(
            robot_model_cfg.robot_description_path
        )

    robot_model = toco.models.RobotModelPinocchio(
        urdf_filename=robot_description_path,
        ee_link_name=robot_model_cfg.ee_link_name
    )

    environment = Environment(
        robot_model_cfg=robot_model_cfg,
        object_centers=object_centers,
        gui=False
    )

    traj_planner = TrajOpt(n_waypoints, start, goal, goal_pose, feat_list, feat_weights, max_iter, environment)

    traj = traj_planner.replan(feat_weights, T, timestep)   # returns Trajectory(waypts, waypts_time) object

    joint_pos_trajectory = traj.waypts
    waypt_times = traj.waypts_time
    print(joint_pos_trajectory)
    print(waypt_times)

    # Calculate joint velocity trajectory
    joint_vel_trajectory = compute_joint_velocities(joint_pos_trajectory, timestep)
    print(joint_vel_trajectory)

    joint_pos_trajectory = to_tensor(joint_pos_trajectory)
    joint_vel_trajectory = to_tensor(joint_vel_trajectory)

    print(joint_pos_trajectory)
    print(joint_vel_trajectory)

    policy = PathFollower(
        joint_pos_trajectory=joint_pos_trajectory,
        joint_vel_trajectory=joint_vel_trajectory,
        Kq=default_kq,
        Kqd=default_kqd,
        Kx=default_kx,
        Kxd=default_kxd,
        robot_model=robot_model,
        timestep=timestep
    )

    # Run policy
    print("\nRunning path follower policy ...\n")
    state_log = robot.send_torch_policy(policy)

    print(environment.get_current_joint_pos())
    print(environment.get_current_joint_vel())

if __name__ == "__main__":
    main()