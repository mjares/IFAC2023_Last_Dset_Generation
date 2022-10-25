import os
import sys
from pathlib import Path
import pickle

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# Adding local libs to path
parentFolder = Path(os.getcwd()).parent.parent.parent
sys.path.append(str(parentFolder) + '\\Z. Local mods')

# Local libs
import quadcopter
import Controller
from Helpers_Generic import set_random_square_path, plot_3d_trajectory, plot_features, LowLvlPID
from TrajectoryDataset import TrajectoryDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

### Inits ###
no_datasets = 2000
starttime_range = [500, 1500]
endtime = 31000
verbose = True
show = False
# Operating Modes
operating_mode_list = ['Nominal', 'Rotor']  # 'Rotor', 'Nominal', 'AttNoise'
no_files = len(operating_mode_list)

### Trajectory tracking ###
for ff in range(no_files):
    saveFilename = f'Dataset_Ctrl_LOE_Nom_{operating_mode_list[ff]}_RandomXYPath_EpCount_{no_datasets}_{ff + 1}.ds'

    if verbose:
        print(saveFilename)

    # Operating Modes
    if operating_mode_list[ff] == 'Nominal':
        operating_modes = {'Nominal': 0}
    elif operating_mode_list[ff] == 'Rotor':
        operating_modes = {'Rotor': [0.01, 0.30]}
    elif operating_mode_list[ff] == 'AttNoiseU':
        operating_modes = {'AttNoise': [0.84, 1.2]}
    elif operating_mode_list[ff] == 'AttNoiseK':
        operating_modes = {'AttNoise': [0.01, 0.84]}

    dataset_list = []

    for ii in range(no_datasets):
        print(f'Sample: {ii + 1}')
        rng = default_rng()
        starttime = rng.integers(starttime_range[0], starttime_range[1])
        total_steps = []
        trajectories = []

        # Make objects for quadcopter
        QUADCOPTER = {
            str(1): {'position': [0, 0, 5], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                     'weight': 1.2}}
        quad = quadcopter.Quadcopter(QUADCOPTER)
        Quads = {str(1): QUADCOPTER[str(1)]}

        # Create blended controller and link it to quadcopter object
        BLENDED_CONTROLLER_PARAMETERS = {
            'Motor_limits': [0, 9000],
            'Tilt_limits': [-10, 10],
            'Yaw_Control_Limits': [-900, 900],
            'Z_XY_offset': 500,
            'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
            'Linear_To_Angular_Scaler': [1, 1, 0],
            'Yaw_Rate_Scaler': 0.18,
            'Angular_PID': LowLvlPID.LOE.value,
            'Angular_PID2': LowLvlPID.Wnd.value,  # The Wind controller is equivalent to the Nominal controller.
             }

        ctrl = Controller.Blended_PID_Controller(quad.get_state,
                                                 quad.set_motor_speeds, quad.get_motor_speeds,
                                                 quad.stepQuad, quad.set_motor_faults, quad.setWind, quad.setNormalWind,
                                                 params=BLENDED_CONTROLLER_PARAMETERS)
        ctrl.set_controller("Uniform")
        goals, safe_region, total_waypoints = set_random_square_path(z=[5, 5, 5, 5, 5, 5, 5, 5])
        currentWaypoint = 0
        ctrl.update_target(goals[currentWaypoint], safe_region[currentWaypoint])

        # Operating Mode
        no_modes = len(operating_modes)
        modes_list = sorted(operating_modes)  # Sorting the key list guarantees consistency
        rng = default_rng()
        rand_mode = rng.integers(0, no_modes)
        rand_key = modes_list[rand_mode]
        ctrl.set_fault_mode(rand_key)
        fault_type = rand_key
        # Setting up selected mode
        if rand_key == 'Nominal':
            fault_mag = 0
            rotor = -1
            # Verbose
            if verbose:
                print('Operating Mode: ', rand_key)  # No changes needed for nominal conditions
        elif rand_key == 'Rotor':
            # Randomly generating anomaly magnitude
            rand_coeff = rng.uniform()  # **** Consider including ranges
            # Getting loe range from operating modes dict
            mode_range = operating_modes[rand_key]
            fault_mag = (mode_range[1] - mode_range[0]) * rand_coeff + mode_range[0]
            # Setting LOE values on a single rotor
            faults = [0, 0, 0, 0]
            rotor = rng.integers(0, 4)  # Random Rotor Selection
            faults[rotor] = fault_mag
            ctrl.set_motor_fault(faults)
            ctrl.set_fault_time(starttime, endtime)
            # Verbose
            if verbose:
                print('LOE (%): ', fault_mag)
        elif rand_key == 'AttNoise':
            rotor = -1
            # Randomly generating anomaly magnitude
            rand_coeff = rng.uniform()  # **** Consider including ranges
            # Getting position noise range from operating modes dict
            mode_range = operating_modes[rand_key]
            fault_mag = (mode_range[1] - mode_range[0]) * rand_coeff + mode_range[0]
            ctrl.set_attitude_sensor_noise(fault_mag)
            ctrl.set_fault_time(starttime, endtime)
            # Verbose
            if verbose:
                print('Attitude Noise (rad): ', fault_mag)

        # Trajectory Tracking Initialization
        done = False
        stableDone = False
        stepcount = 0
        stableAtGoal = 0
        state_log = []
        obs_log = []
        weight_log = []
        motor_log = []
        error_log = []
        goal_log = []

        while not done:
            stepcount += 1

            obs, weight = ctrl.update(return_weight=True)  # No action needed for Random Uniform Control.
            state, control = ctrl.get_full_observation()
            # Logs
            obs_log.append(obs)
            state_log.append(state)
            motor_log.append(control)
            weight_log.append(weight)
            goal_log.append(goals[currentWaypoint])
            error_log.append(ctrl.get_distance_to_opt())

            if stepcount > 10000:  # Max stepcount reached
                print('Not Stable at Goal')
                done = True
                trajectories = ctrl.get_trajectory()
                total_steps.append(ctrl.get_total_steps())

            if ctrl.is_at_pos(goals[currentWaypoint]):  # If the controller has reached the waypoint
                if currentWaypoint < total_waypoints - 1:  # If not final goal
                    currentWaypoint += 1  # Next waypoint
                    ctrl.update_target(goals[currentWaypoint], safe_region[currentWaypoint-1])
                else:  # If final goal
                    stableAtGoal += 1  # Number of timesteps spent within final goal
                    if stableAtGoal > 100:
                        if verbose:
                            print('Stable at Goal')
                        done = True
                        stableDone = True
                        trajectories = ctrl.get_trajectory()
                        total_steps.append(ctrl.get_total_steps())
            else:
                stableAtGoal = 0

        ### Plotting
        trajectories = np.array(trajectories)
        if show:
            plot_3d_trajectory(trajectories, safe_region, fault_type, show=False)
            plt.figure()
            plt.plot(weight_log)
            plot_features(np.array(obs_log), starttime)

        ## To Hildensia Dataset
        dataset = TrajectoryDataset()
        dataset.observations = {'state': np.array(state_log), 'motor_command': np.array(motor_log)}
        dataset.weight = np.array(weight_log)
        dataset.safezone_error = np.array(error_log)
        dataset.goal = np.array(goal_log)
        dataset.safe_region = safe_region
        dataset.trajectories = trajectories
        dataset.conditions = {'Fault': fault_type, 'Mag': fault_mag, 'Rotor': rotor, 'Ctrl': 'Uniform LOE-Nominal'}
        dataset.stable_at_goal = stableDone
        dataset_list.append(dataset)

    if show:
        plt.show()

    datasetFile = open(saveFilename, 'wb')  # Creating file to write
    pickle.dump(dataset_list, datasetFile)
    datasetFile.close()
    print('\n')
