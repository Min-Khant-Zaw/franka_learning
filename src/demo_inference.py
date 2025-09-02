import math
import sys, os
import glob

from planners.trajopt_planner import TrajOpt
from learners.demo_learner import DemoLearner
from utils.my_pybullet_utils import *
from utils.environment import Environment

import torchcontrol as toco
from polymetis import RobotInterface
from polymetis.utils.data_dir import get_full_path_to_urdf

import numpy as np
import pickle
import hydra

@hydra.main(config_path="../config", config_name="demo_inference")
def main(cfg):
    # ----- General Setup ----- #
    feat_list = cfg.setup.feat_list
    demo_spec = cfg.setup.demo_spec
    object_centers = cfg.setup.object_centers

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

    # Learner setup
    constants = {}
    constants["trajs_path"] = cfg.learner.trajs_path
    constants["betas_list"] = cfg.learner.betas_list
    constants["weight_vals"] = cfg.learner.weight_vals
    constants["FEAT_RANGE"] = cfg.learner.FEAT_RANGE
    demo_learner = DemoLearner(feat_list, environment, constants)

    if demo_spec == "simulate":
        # Initialize robot interface
        robot = RobotInterface(
            ip_address=cfg.ip
        )

        # Task setup
        start = cfg.sim.task.start
        goal = cfg.sim.task.goal
        goal_pose = cfg.sim.goal_pose
        T = cfg.sim.task.T
        hz = robot.metadata.hz
        num_steps = int(T * hz)
        timestep = T / num_steps
        feat_weights = cfg.sim.task.feat_weights
        
        # Planner Setup
        planner_type = cfg.sim.planner.type
        if planner_type == "trajopt":
            max_iter = cfg.sim.planner.max_iter
            n_waypoints = cfg.planner.n_waypoints
            
            # Initialize planner and compute trajectory simulation.
            traj_planner = TrajOpt(n_waypoints, start, goal, feat_list, feat_weights, max_iter, environment, goal_pose)
        else:
            raise Exception(f'Planner {planner_type} not implemented.\n')
        
        traj = traj_planner.replan(feat_weights, T, timestep)

        # Downsample trajectory for visualization
        viz_traj = traj.downsample(100)
        viz_traj_waypts = viz_traj.waypts

        # Visualize trajectory
        visualizeTraj(environment, viz_traj_waypts, radius=0.02, color=[0, 0, 1, 1])

    else:
        data_path = cfg.setup.demo_dir
        data_str = demo_spec.split("_")
        data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')) + data_path
        if data_str[0] == "all":
            file_str = data_dir + '/*task{}*.p'.format(data_str[1])
        else:
            file_str = data_dir + "/demo_ID{}_task{}*.p".format(data_str[0], data_str[1])

        traj = [pickle.load(open(demo_file, "rb")) for demo_file in glob.glob(file_str)]
    
    demo_learner.learn_weights(traj)

if __name__ == "__main__":
    main()