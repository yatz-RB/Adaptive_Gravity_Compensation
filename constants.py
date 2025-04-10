import pathlib

import numpy as np

### Task parameters
DATA_DIR = 'data'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
}

### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = {
    'HEAD': ['neck_joint'],
    'LEFT': ['left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6', 'left_finger1_joint', 'left_finger2_joint'],
    'RIGHT': ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_finger1_joint', 'right_finger2_joint'],
}
START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/'

# Arm position limits
JOINT_LIMITS = ((-3.11, 3.11), (-2.27, 2.27), (-2.36, 2.36), (-3.11, 3.11), (-2.23, 2.23), (-6.28, 6.28))
ARM_POSITION_MAX = [3.106687, 2.268929, 2.356196, 3.106687, 2.234022, 6.283188]
ARM_POSITION_MIN = [-3.106687, -2.268929, -2.356196, -3.106687, -2.234022, -6.283188]
GRIPPER_MOTOR_MAX = 100
GRIPPER_MOTOR_MIN = 0

# MDH fixed constants
MDH = {
    'alpha': [0, np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2],
    'a': [0, 0, 0.256, 0, 0, 0],
    'd': [0.2405, 0, 0, 0.21, 0, 0.144],
    'offset': [0, np.pi / 2, np.pi / 2, 0, 0, 0]
}
T_tool = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.2],
    [0, 0, 0, 1]
])

# Dynamic constants
DYNAMICS = {
    'm': [1.51, 1.653, 0.726, 0.671, 0.647, 0.107],
    'x': [0.000491, 0.183722, 0.000029, 0.000007, 0.000032, -0.000506],
    'y': [0.007803, 0.000103, -0.090105, -0.009486, -0.083769, -0.000255],
    'z': [-0.010744, -0.001665, 0.004039, -0.008041, 0.002326, -0.010801],
    'l_xx': [0.002928466, 0.001711553, 0.007259884, 0.000794014, 0.005375604, 5.0918e-05],
    'l_xy': [-3.263e-05, -3.8271e-05, 2.994e-06, -8.21e-07, 2.665e-06, -3.136e-06],
    'l_xz': [-5.816e-06, 0.00231491, -3.14e-07, -6.55e-07, -3.04e-07, -6.99e-07],
    'l_yy': [0.00250635, 0.070514722, 0.000371872, 0.000596235, 0.000285265, 4.742e-05],
    'l_yz': [4.7925e-05, 6.507e-06, 4.4451e-05, -3.4785e-05, 1.4235e-05, 3.88e-07],
    'l_zz': [0.001756017, 0.070036186, 0.007228758, 0.000486228, 0.005359769, 6.035e-05]
}

# base link
PUBLIC_BASE_LINK = 'shoulder_link'
LEFT_BASE = 'left_base'
RIGHT_BASE = 'right_base'
