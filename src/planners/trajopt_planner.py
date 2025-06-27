import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint

from utils.trajectory import Trajectory

class TrajOpt(object):
    def __init__(self, n_waypoints, start, goal, goal_pose, feat_list, max_iter, environment, traj_seed=None):
        # Set hyperparameters
        self.n_joints = len(start)
        self.start = start
        self.goal = goal
        self.goal_pose = goal_pose
        self.feat_list = feat_list
        self.num_features = len(self.feat_list)

        # Initilize weights for each feature
        self.weights = [0.0] * self.num_features

        # Variables for trajpot parameters
        self.n_waypoints = n_waypoints
        self.MAX_ITER = max_iter

        # Set Pybullet environment
        self.environment = environment

        # Create initial trajectory
        if traj_seed is None:
            print("Using straight line initialization!")
            self.xi0 = np.zeros((self.n_waypoints, self.n_joints))  # 5x7
            for idx in range(self.n_waypoints):
                self.xi0[idx,:] = self.start + idx/(self.n_waypoints - 1.0) * (self.goal - self.start)
            self.xi0 = self.xi0.reshape(-1) # flatten initial trajectory into 1D array for scipy minimize (1x35)
        else:
            print("Using trajectory seed initialization!")
            self.xi0 = traj_seed

        # Create start point equality constraint
        self.B = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))    # 7x35
        for idx in range(self.n_joints):
            self.B[idx,idx] = 1
        self.start_constraint = LinearConstraint(self.B, self.start, self.start)

        # Create goal pose constraint if given
        if self.goal_pose is not None:
            self.goal_xyz = np.array(self.goal_pose)
            self.goal_constraint = NonlinearConstraint(self.goal_pose_constraint, lb=self.goal_xyz, ub=self.goal_xyz)
        else:
            self.B_goal = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))
            for idx in range(self.n_joints):
                self.B_goal[idx, (self.n_waypoints - 1) * self.n_joints + idx] = 1
            self.goal_constraint = LinearConstraint(self.B_goal, self.goal, self.goal)

        # Create table constraint
        self.table_constraint = NonlinearConstraint(self.table_constraint_batch, 0, np.inf)

    # Constraint functions
    def table_constraint_batch(self, xi: np.ndarray):
        """
		Constrains z-axis of robot's end-effector to always be 
		above the table by taking the trajectory as the input.
		"""
        xi = xi.reshape((self.n_waypoints, self.n_joints))
        return np.array([self.environment.table_constraint(xi[t]) for t in range(1, self.n_waypoints - 1)])
    
    def goal_pose_constraint(self, xi: np.ndarray):
        """
        Constrains the goal pose of the end-effector to be the same
        as the given goal pose.
        """
        xi = xi.reshape((self.n_waypoints, self.n_joints))
        last_waypt = xi[-1]
        ee_position, _ = self.environment.compute_forward_kinematics(self, last_waypt)
        return ee_position

    # Cost functions
    def efficiency_cost(self, waypt):
        """
		Computes the total efficiency cost
		---
		input two consecutive waypoints, output scalar cost
		"""
        prev_waypt = waypt[0:7]
        current_waypt = waypt[7:14]
        feature = self.environment.efficiency_features(current_waypt, prev_waypt)
        feature_idx = self.feat_list.index("efficiency")
        return feature * self.weights[feature_idx]
    
    def origin_cost(self, waypt):
        """
		Computes the total distance from EE to base of robot cost.
		---
		input waypoint, output scalar cost
		"""
        feature = self.environment.origin_features(waypt)
        feature_idx = self.feat_list.index('origin')
        return feature * self.weights[feature_idx]
    
    def table_cost(self, waypt):
        """
        Computes the total distance to table cost.
        ---
		input waypoint, output scalar cost
        """
        feature = self.environment.table_features(waypt)
        feature_idx = self.feat_list.index("table")
        return feature * self.weights[feature_idx]
    
    def coffee_cost(self, waypt):
        """
        Computes the total coffee (EE orientation cost)
        ---
        input waypoint, output scalar cost
        """
        feature = self.environment.coffee_features(waypt)
        feature_idx = self.feat_list.index("coffee")
        return feature * self.weights[feature_idx]
    
    def laptop_cost(self, waypt):
        """
        Computes the total distance to laptop cost
		---
		input waypoint, output scalar cost
		"""
        prev_waypt = waypt[0:7]
        curr_waypt = waypt[7:14]
        feature = self.environment.laptop_features(curr_waypt,prev_waypt)
        feature_idx = self.feat_list.index('laptop')
        return feature * self.weights[feature_idx] * np.linalg.norm(curr_waypt - prev_waypt)

    def human_cost(self, waypt):
        """
		Computes the total distance to human cost.
		---
		input waypoint, output scalar cost
		"""
        prev_waypt = waypt[0:7]
        curr_waypt = waypt[7:14]
        feature = self.environment.human_features(curr_waypt,prev_waypt)
        feature_idx = self.feat_list.index('human')
        return feature * self.weights[feature_idx] * np.linalg.norm(curr_waypt - prev_waypt)

    # Problem specific cost function
    def trajcost(self, xi):
        xi = xi.reshape((self.n_waypoints, self.n_joints))    # 5x7
        cost_total = 0

        for idx in range(1, self.n_waypoints):
            # state cost functions
            if 'coffee' in self.feat_list:
                cost_total += self.coffee_cost(xi[idx])
            if 'table' in self.feat_list:
                cost_total += self.table_cost(xi[idx])
            if 'origin' in self.feat_list:
                cost_total += self.origin_cost(xi[idx])
            if 'laptop' in self.feat_list:
                cost_total += self.laptop_cost(xi[idx - 1] + xi[idx])
            if 'human' in self.feat_list:
                cost_total += self.human_cost(xi[idx - 1] + xi[idx])
            if 'efficiency' in self.feat_list:
                cost_total += self.efficiency_cost(xi[idx - 1] + xi[idx])

        return cost_total

    # Use scipy optimizer to get optimal trajectory
    def optimize(self, method='SLSQP'):
        start_t = time.time()
        res = minimize(self.trajcost, self.xi0, method=method, constraints=[self.start_constraint, self.table_constraint, self.goal_constraint])
        xi = res.x.reshape(self.n_waypoints,self.n_joints)
        return xi, res, time.time() - start_t
    
    def replan(self, weights, T, timestep):
        """
        Replan the trajectory from start to goal given weights.
        ---
        Parameters:
            start -- Start position
            goal -- Goal position.
            goal_pose -- Goal pose (optional: can be None).
            weights -- Weights used for the planning objective.
            T [float] -- Time horizon for the desired trajectory.
            timestep [float] -- Frequency of waypoints in desired trajectory.
        Returns:
            traj [Trajectory] -- The optimal trajectory satisfying the arguments.
        """
        assert weights is not None, "The weights vector is empty. Cannot plan without a cost preference."
        self.weights = weights

        waypts = self.optimize(method='SLSQP')
        waypts_time = np.linspace(0.0, T, self.n_waypoints)
        traj = Trajectory(waypts, waypts_time)
        return traj.upsample(int(T/timestep) + 1)