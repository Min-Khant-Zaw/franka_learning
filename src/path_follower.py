from typing import Dict, List

import torch

import torchcontrol as toco
from torchcontrol.utils.tensor_utils import to_tensor, stack_trajectory

from planners.trajopt_planner import TrajOpt

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
        assert self.joint_pos_trajectory == self.joint_vel_trajectory

        # Control modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            robot_model=robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

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

        return {"joint_torques": torque_output}