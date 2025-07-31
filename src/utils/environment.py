import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import torch

from typing import List, Dict
import os
import logging

from omegaconf import DictConfig

from polysim.envs import AbstractControlledEnv
from polymetis.utils.data_dir import get_full_path_to_urdf
from torchcontrol.utils.tensor_utils import to_tensor

from utils.my_pybullet_utils import *

from tf.transformations import euler_from_quaternion

log = logging.getLogger(__name__)

class Environment(AbstractControlledEnv):
    def __init__(
            self,
            robot_model_cfg: DictConfig,
            object_centers: Dict[str, List[float]],
            robot_model: torch.nn.Module,
            gui: bool,
            use_grav_comp: bool = True,
            gravity: float = 9.81,
            extract_config_from_rdf=False,
    ):
        # Get objects
        self.object_centers = object_centers

        # Create environment
        self.gui = gui
        self.use_grav_comp = use_grav_comp

        self.robot_model_cfg = robot_model_cfg
        self.robot_model = robot_model

        # get the absolute path to urdf of the robot
        self.robot_description_path = get_full_path_to_urdf(
            self.robot_model_cfg.robot_description_path
        )

        # get the data from robot_model_cfg
        self.controlled_joints = self.robot_model_cfg.controlled_joints
        self.num_dofs = self.robot_model_cfg.num_dofs
        assert len(self.controlled_joints) == self.num_dofs
        self.ee_link_idx = self.robot_model_cfg.ee_link_idx
        self.ee_link_name = self.robot_model_cfg.ee_link_name
        self.rest_pose = self.robot_model_cfg.rest_pose
        self.joint_limits_low = np.array(self.robot_model_cfg.joint_limits_low)
        self.joint_limits_high = np.array(self.robot_model_cfg.joint_limits_high)
        if self.robot_model_cfg.joint_damping is None:
            self.joint_damping = None
        else:
            self.joint_damping = np.array(self.robot_model_cfg.joint_damping)
        if self.robot_model_cfg.torque_limits is None:
            self.torque_limits = np.inf * np.ones(self.n_dofs)
        else:
            self.torque_limits = np.array(self.robot_model_cfg.torque_limits)

        # Initialize PyBullet simulation
        if self.gui:
            self.sim = BulletClient(connection_mode=p.GUI)
        else:
            self.sim = BulletClient(connection_mode=p.DIRECT)

        self.sim.setGravity(0, 0, -gravity)

        # Load robot from urdf or sdf
        ext = os.path.splitext(self.robot_description_path)[-1][1:]
        if ext == "urdf":
            self.world_id, self.robot_id = self.load_robot_description_from_urdf(
                self.robot_description_path, self.sim
            )
        elif ext == "sdf":
            self.world_id, self.robot_id = self.load_robot_description_from_sdf(
                self.robot_description_path, self.sim
            )
        else:
            raise Exception(f"Unknown robot definition extension {ext}!")
        
        # Add objects
        addTable(tablePos=[0.5, 0.0, 0.0])
        addRobotStand(standPos=[0.5, 0.9, 0.0])
        if "HUMAN_CENTER" in self.object_centers:
            addHuman(self.object_centers)
        if "LAPTOP_CENTER" in self.object_centers:
            addLaptop(self.object_centers)
        
        # Debug utility to print out joint configuration information directly from the URDF/SDF
        if extract_config_from_rdf:
            log.info("************ CONFIG INFO ************")
            num_joints = self.sim.getNumJoints(self.robot_id)
            for i in range(num_joints):
                # joint_info tuple structure described in PyBullet docs:
                # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.la294ocbo43o
                joint_info = self.sim.getJointInfo(self.robot_id, i)
                log.info("Joint {}".format(joint_info[1].decode("utf-8")))  # jointName
                log.info("\tLimit low : {}".format(joint_info[8]))  # jointLowerLimit
                log.info("\tLimit High: {}".format(joint_info[9]))  # jointUpperLimit
                log.info("\tJoint Damping: {}".format(joint_info[6]))  # jointDamping
            log.info("*************************************")

        # Enable torque control
        self.sim.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(self.num_dofs),
        )

        # Initialize variables
        self.prev_torques_commanded = np.zeros(self.num_dofs)
        self.prev_torques_applied = np.zeros(self.num_dofs)
        self.prev_torques_measured = np.zeros(self.num_dofs)
        self.prev_torques_external = np.zeros(self.num_dofs)

    @staticmethod
    def load_robot_description_from_urdf(abs_urdf_path: str, sim: BulletClient):
        """
        Loads a URDF file into the simulation.
        """
        log.info("loading urdf file: {}".format(abs_urdf_path))
        robot_id = sim.loadURDF(
            abs_urdf_path,          # might have to change this (different from the urdf from pybullet_data)
            basePosition=[0.5, 0.75, 0.7],
            baseOrientation=[0.0, 0.0, -0.707, 0.707],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        world_id = p.loadURDF("plane.urdf", [0.0, 0.0, 0.0])
        return world_id, robot_id
    
    @staticmethod
    def load_robot_description_from_sdf(abs_urdf_path: str, sim: BulletClient):
        """
        Loads a SDF file into the simulation.
        """
        log.info("loading sdf file: {}".format(abs_urdf_path))
        robot_id = sim.loadSDF(
            abs_urdf_path,
        )[0]

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        world_id = p.loadURDF("plane.urdf", [0.0, 0.0, 0.0])
        return world_id, robot_id

    def reset(self, joint_pos: List[float] = None, joint_vel: List[float] = None):
        """
        Resets simulation to the given pose, or if not given, the default rest pose
        """
        if joint_pos is None:
            joint_pos = self.rest_pose
        if joint_vel is None:
            joint_vel = [0 for _ in joint_pos]

        for i in range(self.num_dofs):
            self.sim.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.controlled_joints[i],
                targetValue=joint_pos[i],
                targetVelocity=joint_vel[i]
            )

    def get_num_dofs(self):
        """
        Get the number of degrees of freedom for controlling the simulation.

        Returns:
          int: Number of control input dimensions
        """
        return self.num_dofs

    def get_current_joint_pos_vel(self):
        """""
        Returns (current joint position, current joint velocity) as a tuple of NumPy arrays

        Returns:
          np.ndarray: Joint positions
          np.ndarray: Joint velocities
        """
        curr_joint_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints
        )
        curr_joint_pos = [curr_joint_states[i][0] for i in range(self.num_dofs)]
        curr_joint_vel = [curr_joint_states[i][1] for i in range(self.num_dofs)]
        return np.array(curr_joint_pos), np.array(curr_joint_vel)
    
    def get_current_joint_pos(self):
        """
        Returns current joint position as a NumPy array.
        """
        return self.get_current_joint_pos_vel()[0]

    def get_current_joint_vel(self):
        """
        Returns current joint velocity as a NumPy array.
        """
        return self.get_current_joint_pos_vel()[1]

    def get_current_joint_torques(self):
        """
        Returns:
          np.ndarray: Torques received from apply_joint_torques
          np.ndarray: Torques sent to robot (e.g. after clipping)
          np.ndarray: Torques generated by the actuators (e.g. after grav comp)
          np.ndarray: Torques exerted onto the robot
        """
        return (
            self.prev_torques_commanded,
            self.prev_torques_applied,
            self.prev_torques_measured,
            self.prev_torques_external
        )

    def apply_joint_torques(self, torques: np.ndarray):
        """
        input:
          np.ndarray: Desired torques

        Returns:
          np.ndarray: Final applied torques
        """
        assert isinstance(torques, np.ndarray)
        self.prev_torques_commanded = torques

        # Do not mutate original array!
        applied_torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        self.prev_torques_applied = applied_torques.copy()

        # Add gravity compensation if enabled
        if self.use_grav_comp:
            curr_joint_pos = self.get_current_joint_pos()
            grav_comp_torques = self.compute_inverse_dynamics(
                joint_pos=curr_joint_pos,
                joint_vel=[0] * self.num_dofs,
                joint_acc=[0] * self.num_dofs
            )
            applied_torques += grav_comp_torques
        self.prev_torques_applied = applied_torques.copy()

        self.sim.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joints,
            p.TORQUE_CONTROL,
            forces=applied_torques,
        )

        self.sim.stepSimulation()

        return applied_torques
    
    def compute_forward_kinematics(self, joint_pos: List[float] = None):
        """
        Computes forward kinematics.

        Warning:
            Uses PyBullet forward kinematics by resetting to the given joint position (or, if not given,
            the rest position). Therefore, drastically modifies simulation state!

        Args:
            joint_pos: Joint positions for which to compute forward kinematics.

        Returns:
            np.ndarray: 3-dimensional end-effector position

            np.ndarray: 4-dimensional end-effector orientation as quaternion

        """
        if joint_pos != None:
            # log.warning(
            #     "Resetting PyBullet simulation to given joint_pos to compute forward kinematics!"
            # )
            self.reset(joint_pos)
        link_state = self.sim.getLinkState(
            self.robot_id,
            self.ee_link_idx,
            computeForwardKinematics=True,
        )
        ee_position = np.array(link_state[4])
        ee_orient_quaternion = np.array(link_state[5])

        return ee_position, ee_orient_quaternion

    # def compute_forward_kinematics(self, joint_pos: List[float]):
    #     """
    #     Computes forward kinematics.

    #     Args:
    #         joint_pos: Joint positions for which to compute forward kinematics.

    #     Returns:
    #         np.ndarray: 3-dimensional end-effector position

    #         np.ndarray: 4-dimensional end-effector orientation as quaternion

    #     """
    #     joint_positions = to_tensor(joint_pos)

    #     result = self.robot_model.forward_kinematics(joint_positions)

    #     ee_position = result[0].numpy()
    #     ee_orient_quaternion = result[1].numpy()

    #     return ee_position, ee_orient_quaternion

    def compute_inverse_kinematics(
        self, target_position: List[float], target_orientation: List[float]=None
    ):
        """
        Computes inverse kinematics.

        Args:
            target_position: 3-dimensional desired end-effector position.
            target_orientation: 4-dimensional desired end-effector orientation as quaternion.

        Returns:
            np.ndarray: Joint position satisfying the given target end-effector position/orientation.

        """
        if isinstance(target_position, np.ndarray):
            target_position = target_position.tolist()
        if isinstance(target_orientation, np.ndarray):
            target_orientation = target_orientation.tolist()
        # ik_kwargs = dict(
        #     bodyUniqueId=self.robot_id,
        #     endEffectorLinkIndex=self.ee_link_idx,
        #     targetPosition=target_position,
        #     targetOrientation=target_orientation,
        #     upperLimits=self.joint_limits_high.tolist(),
        #     lowerLimits=self.joint_limits_low.tolist(),
        # )
        # if self.joint_damping is not None:
        #     ik_kwargs["joint_damping"] = self.joint_damping.tolist()
        desired_joint_pos = self.sim.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_idx,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            upperLimits=self.joint_limits_high.tolist(),
            lowerLimits=self.joint_limits_low.tolist(),
            jointDamping=self.joint_damping.tolist()if self.joint_damping is not None else None
        )
        return np.array(desired_joint_pos)
    
    def compute_inverse_dynamics(
        self, joint_pos: np.ndarray, joint_vel: np.ndarray, joint_acc: np.ndarray
    ):
        """
        Computes inverse dynamics by returning the torques necessary to get the desired accelerations
        at the given joint position and velocity.
        """
        torques = self.sim.calculateInverseDynamics(
            self.robot_id, list(joint_pos), list(joint_vel), list(joint_acc)
        )
        return np.asarray(torques)

    def set_robot_state(self, robot_state):
        """
        input:
          RobotState: Desired state of robot

        output:
          RobotState: Resulted state of robot
        """
        req_joint_pos = robot_state.joint_positions
        req_joint_vel = robot_state.joint_velocities
        assert len(req_joint_pos) == len(req_joint_vel) == self.num_dofs
        for i in range(self.num_dofs):
            self.sim.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                targetValue=req_joint_pos[i],
                targetVelocity=req_joint_vel[i],
            )

        grav_comp_torques = self.compute_inverse_dynamics(
            joint_pos=req_joint_pos,
            joint_vel=[0] * self.num_dofs,
            joint_acc=[0] * self.num_dofs,
        )

        # bug in pybullet requires many steps after reset so that subsequent forward sim calls work correctly
        for _ in range(100):
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.controlled_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=grav_comp_torques,
            )
            self.sim.stepSimulation()

        joint_states = self.sim.getJointStates(
            bodyUniqueId=self.robot_id,
            jointIndices=self.controlled_joints,
        )
        return joint_states
    
    # ---- Custom environmental features ---- #
    def featurize(self, waypts, feat_list):
        """
		Computes the user-defined features for a given trajectory.
		---
		input trajectory waypoints, output list of feature values
		"""
        num_features = len(feat_list)
        features = [[0.0 for _ in range(len(waypts)-1)] for _ in range(0, num_features)]

        for index in range(len(waypts)-1):
            for feat in range(num_features):
                if feat_list[feat] == 'table':
                    features[feat][index] = self.table_features(waypts[index+1])
                elif feat_list[feat] == 'coffee':
                    features[feat][index] = self.coffee_features(waypts[index+1])
                elif feat_list[feat] == 'human':
                    features[feat][index] = self.human_features(waypts[index+1],waypts[index])
                elif feat_list[feat] == 'laptop':
                    features[feat][index] = self.laptop_features(waypts[index+1],waypts[index])
                elif feat_list[feat] == 'origin':
                    features[feat][index] = self.origin_features(waypts[index+1])
                elif feat_list[feat] == 'efficiency':
                    features[feat][index] = self.efficiency_features(waypts[index+1],waypts[index])
        return features
    
    # Efficiency
    def efficiency_features(self, waypt, prev_waypt):
        """
	    Computes efficiency feature for waypoint, confirmed to match trajopt.
	    ---
	    input two consecutive waypoints, output scalar feature
	    """
        return np.linalg.norm(waypt - prev_waypt)**2
        
    # Distance to Robot Base
    def origin_features(self, waypt):
        """
		Computes the total feature value over waypoints based on 
		distance from the base of the robot to its end-effector.
		---
		input waypoint, output scalar feature
		"""
        waypt = waypt.tolist()
        ee_position, _ = self.compute_forward_kinematics(waypt)
        ee_position = np.array(ee_position)
        return np.linalg.norm(ee_position)
    
    # Distance to Table
    def table_features(self, waypt):
        """
		Computes the total feature value over waypoints based on 
		z-axis distance to table.
		---
		input waypoint, output scalar feature
		"""
        waypt = waypt.tolist()
        ee_position, _ = self.compute_forward_kinematics(waypt)
        ee_coord_z = ee_position[2]
        return ee_coord_z
    
    # Coffee (or z-orientation of end-effector)
    def coffee_features(self, waypt):
        """
		Computes the distance to table feature value for waypoint
		by checking if the EE is oriented vertically according to pitch.
		---
		input waypoint, output scalar feature
		"""
        waypt = waypt.tolist()
        _, ee_orientation = self.compute_forward_kinematics(waypt)
        [roll, pitch, yaw] = euler_from_quaternion(ee_orientation)
        return abs(pitch)

    # Distance to laptop
    def laptop_features(self, waypt, prev_waypt):
        """
		Computes laptop feature value over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions.
		---
		input two consecutive waypoints, output scalar feature
		"""
        feature = 0.0
        NUM_STEPS = 4
        for step in range(NUM_STEPS):
            inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
            feature += self.laptop_dist(inter_waypt)
        return feature
    
    def laptop_dist(self, waypt):
        """
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
        waypt = waypt.tolist()
        ee_position, _ = self.compute_forward_kinematics(waypt)
        ee_coord_xy = ee_position[0:2]
        ee_coord_xy = np.array(ee_coord_xy)
        laptop_xy = np.array(self.object_centers["LAPTOP_CENTER"][0:2])
        dist = np.linalg.norm(ee_coord_xy - laptop_xy) - 0.4
        if dist > 0:
            return 0
        return -dist
    
    # Distance to human
    def human_features(self, waypt, prev_waypt):
        """
		Computes laptop feature value over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions.
		---
		input two consecutive waypoints, output scalar feature
		"""
        feature = 0.0
        NUM_STEPS = 4
        for step in range(NUM_STEPS):
            inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
            feature += self.human_dist(inter_waypt)
        return feature
    
    def human_dist(self, waypt):
        """
		Computes distance from end-effector to human in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from human
			+: EE is closer than 0.4 meters to human
		"""
        waypt = waypt.tolist()
        ee_position, _ = self.compute_forward_kinematics(waypt)
        ee_coord_xy = ee_position[0:2]
        ee_coord_xy = np.array(ee_coord_xy)
        laptop_xy = np.array(self.object_centers["HUMAN_CENTER"][0:2])
        dist = np.linalg.norm(ee_coord_xy - laptop_xy) - 0.4
        if dist > 0:
            return 0
        return -dist
    
    # ---- Custom environmental constraints --- #
    def table_constraint(self, waypt):
        """
		Constrains z-axis of robot's end-effector to always be 
		above the table.
		"""
        waypt = waypt.tolist()
        ee_position, _ = self.compute_forward_kinematics(waypt)
        ee_coord_z = ee_position[2]
        return ee_coord_z

    def kill_environment(self):
        """
		Destroys Pybullet thread and environment for clean shutdown.
		"""
        self.sim.disconnect()