import pybullet as p
import time
import pybullet_data
import numpy as np

object_centers = {"HUMAN_CENTER": [-0.5, -0.55, 0.9], "LAPTOP_CENTER": [-0.7929,-0.1,0.0]}

def start_environment(object_centers, direct=False):
    # Connect to physics simulator
    if direct:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    objectID = setup_environment(object_centers)

    p.setGravity(0, 0, -10)
    
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

    return objectID

def setup_environment(object_centers):
    objectID = {}

    # Add the floor
    objectID["plane"] = p.loadURDF("plane.urdf")

    # Add a table
    tablePos = [-0.65, 0.0, 0.0]
    tableOrientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["table"] = p.loadURDF("table/table.urdf", tablePos, tableOrientation, useFixedBase=True)

    # Add a robot support
    # supportPos = [0.0, 0, 0.65]
    # supportOrientation = p.getQuaternionFromEuler([0, 0, 0])
    # objectID["stand"] = p.loadURDF("support.urdf", supportPos, supportOrientation, useFixedBase=True)

    # Add the laptop
    # if "LAPTOP_CENTER" in object_centers:
    #     laptopOrientation = p.getQuaternionFromEuler([0, 0, 0])
    #     objectID["laptop"] = p.loadURDF("laptop.urdf", object_centers["LAPTOP_CENTER"], laptopOrientation, useFixedBase=True)

    # Add the human
    if "HUMAN_CENTER" in object_centers:
        humanOrientation = p.getQuaternionFromEuler([1.5, 0, 1.5])
        objectID["human"] = p.loadURDF("humanoid/humanoid.urdf", object_centers["HUMAN_CENTER"], humanOrientation, globalScaling=0.25, useFixedBase=True)

    # Add Franka Panda arm
    frankaPos = [0, 0, 0.6]
    frankaOrientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["robot"] = p.loadURDF("franka_panda/panda.urdf", frankaPos , frankaOrientation, useFixedBase=True)

    return objectID

start_environment(object_centers)