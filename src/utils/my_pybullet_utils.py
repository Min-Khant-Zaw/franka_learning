import pybullet as p
import time
import pybullet_data
import numpy as np

def addTable():
    tablePos = [0.5, 0.0, 0.0]
    tableOrientation = p.getQuaternionFromEuler([0, 0, 0])
    table_id = p.loadURDF("table/table.urdf", tablePos, tableOrientation, useFixedBase=True)

def addHuman(object_centers):
    humanOrientation = p.getQuaternionFromEuler([1.5, 0, 1.5])
    human_id = p.loadURDF("humanoid/humanoid.urdf", object_centers["HUMAN_CENTER"], humanOrientation, globalScaling=0.25, useFixedBase=True)