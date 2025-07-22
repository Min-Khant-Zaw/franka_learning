from polymetis import RobotInterface

# Initialize robot interface
robot = RobotInterface(
    ip_address="localhost"
)

current_joint_pos = robot.get_joint_positions()
current_joint_vel = robot.get_joint_velocities()
current_ee_pose, current_ee_quat = robot.get_ee_pose()

print(f"Current joint position: {current_joint_pos}\n")
print(f"Current joint velocity: {current_joint_vel}\n")
print(f"Current ee position: {current_ee_pose}\n")
print(f"Current ee orientation in quaternion: {current_ee_quat}\n")