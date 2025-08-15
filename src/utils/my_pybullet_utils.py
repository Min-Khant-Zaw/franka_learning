from typing import List

import pybullet as p
import time
import pybullet_data
import numpy as np

# object_centers = {"HUMAN_CENTER": [0.5, -0.55, 0.9], "LAPTOP_CENTER": [-0.7929, -0.1, 0.0]}

# def start_environment(object_centers, direct=False):
#     # Connect to physics simulator
#     if direct:
#         physicsClient = p.connect(p.DIRECT)
#     else:
#         physicsClient = p.connect(p.GUI)

#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -10)

#     objectID = setup_environment(object_centers)

#     while True:
#         p.stepSimulation()
#         time.sleep(1./240.)

#     return objectID

# def setup_environment(object_centers):
#     objectID = {}

#     # Add the floor
#     objectID["plane"] = p.loadURDF("plane.urdf")

#     # Add a robot support
#     # supportPos = [0.0, 0, 0.65]
#     # supportOrientation = p.getQuaternionFromEuler([0, 0, 0])
#     # objectID["stand"] = p.loadURDF("support.urdf", supportPos, supportOrientation, useFixedBase=True)

#     # Add the laptop
#     # if "LAPTOP_CENTER" in object_centers:
#     #     laptopOrientation = p.getQuaternionFromEuler([0, 0, 0])
#     #     objectID["laptop"] = p.loadURDF("laptop.urdf", object_centers["LAPTOP_CENTER"], laptopOrientation, useFixedBase=True)

#     # Add Franka Panda arm
#     frankaPos = [0, 0, 0.6]
#     frankaOrientation = p.getQuaternionFromEuler([0, 0, 0])
#     objectID["robot"] = p.loadURDF("franka_panda/panda.urdf", frankaPos , frankaOrientation, useFixedBase=True)

#     addTable()
#     addHuman(object_centers)
#     plotSphere([0.7, 0, 1], radius=0.05)

#     return objectID

def addTable(tablePos):
    tableOrientation = p.getQuaternionFromEuler([0, 0, 0])
    table_id = p.loadURDF("table/table.urdf", tablePos, tableOrientation, useFixedBase=True)

def addHuman(object_centers):
    humanOrientation = p.getQuaternionFromEuler([1.5, 0, 1.5])
    human_id = p.loadURDF("humanoid/humanoid.urdf", object_centers["HUMAN_CENTER"], humanOrientation, globalScaling=0.25, useFixedBase=True)

def addRobotStand(standPos):
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.3, 0.3, 0.35],
        collisionFramePosition=[0, 0, 0.35]
    )

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.3, 0.3, 0.35],
        rgbaColor=[0.5, 0.5, 0.5, 1],
        visualFramePosition=[0, 0, 0.35]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=standPos
    )

def addLaptop(object_centers):
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.19, 0.14, 0.125],
        collisionFramePosition=[0, 0, 0.125]
    )

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.19, 0.14, 0.125],
        rgbaColor=[1, 0, 0, 1],
        visualFramePosition=[0, 0, 0.125]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=object_centers["LAPTOP_CENTER"]
    )

def plotSphere(coords, radius=0.02, color=[0, 0, 1, 1]):
    """
	Plots a single sphere in Pybullet centered at coords(x,y,z) location.
	"""
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, 
        radius=radius, 
        rgbaColor=color, 
        visualFramePosition=[0, 0, 0]
    )

    p.createMultiBody(
        baseMass=0,  # mass=0 makes it static
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=coords
    )
    
def visualizeTraj(env, waypoints, radius=0.02, color=[0, 0, 1, 1]):
    """
	Plots the trajectory found or planned.
	"""
    print("Visualizing trajectory.....\n")
    for waypoint in waypoints:
        waypoint = waypoint.tolist()
        ee_position, _ = env.compute_forward_kinematics(waypoint)
        plotSphere(ee_position, radius, color)

# start_environment(object_centers)