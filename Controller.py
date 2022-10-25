import numpy as np
from numpy.random import default_rng
import math
import scipy.stats as stats


class Blended_PID_Controller():
    def __init__(self, get_state, actuate_motors, get_motor_speed, step_quad, set_faults, setWind, ### Must check the difference between setWind and setNormWind
                 setNormWind, params, quad_identifier='1'):
        ## Inits
        self.loe_log = []
        self.controller_type = None
        self.blendDist = None
        # Quadcopter parameters
        self.quad_identifier = quad_identifier
        self.actuate_motors = actuate_motors
        self.set_motor_faults = set_faults
        self.get_state = get_state
        self.step_quad = step_quad
        self.get_motor_speed = get_motor_speed
        self.setWind = setWind
        self.set_normal_wind = setNormWind
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0]/180.0)*3.14, (params['Tilt_limits'][1]/180.0)*3.14]
        # Control parameters
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0]+params['Z_XY_offset'], self.MOTOR_LIMITS[1]-params['Z_XY_offset']]
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        # Position Controller
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        # Attitude Controller 1
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        # Attitude Controller 1
        self.ANGULAR_P2 = params['Angular_PID2']['P']
        self.ANGULAR_I2 = params['Angular_PID2']['I']
        self.ANGULAR_D2 = params['Angular_PID2']['D']
        self.thetai_term2 = 0
        self.phii_term2 = 0
        self.gammai_term2 = 0

        self.total_steps = 0
        self.motor_commands = [0, 0, 0, 0]

        # Fault conditions
        self.FaultMode = "None"
        # Noise
        self.noiseMag = 0
        self.attNoiseMag = 0
        self.x_noise = 0.0
        self.y_noise = 0.0
        self.z_noise = 0.0
        self.theta_noise = 0.0
        self.phi_noise = 0.0
        self.gamma_noise = 0.0
        # Wind
        self.setNormWind = setNormWind
        self.normal_wind = [0, 0, 0]
        self.trajectory = []
        self.trackingErrors = {"Pos_err": 0, "Att_err": 0}
        rng = default_rng()
        self.startfault = rng.integers(500,  2000)
        self.endfault = rng.integers(1500, 3000)
        self.fault_time = [self.startfault, self.endfault]
        # Rotor
        self.motor_faults = [0, 0, 0, 0]
        # Intermittent Rotor
        self.rotor_switching_period = 0
        self.intermittent_rotor_on = False

        self.current_obs = {"x": 0,
                            "y": 0,
                            "z": 0,
                            "phi": 0,
                            "theta": 0, "gamma": 0, "x_err": 0, "y_err": 0,
                            "z_err": 0, "phi_err": 0, "theta_err": 0, "gamma_err": 0}

        # Bounds
        self.safe_bound = []
        self.current_distance_to_opt = 0
        self.safety_margin = 1
        self.goal_safety_margin = 0.5
        self.outsideBounds = False
        self.total_steps_out_of_bounds = 0

        # Setting up blending
        self.mu = 0.5
        self.sigma = 0.1
        self._set_att_blend_dist([self.mu, self.sigma])

        self.target = [0, 0, 0]
        self.yaw_target = 0.0  # Always 0. No Yaw tracking implemented.
        self.set_controller("Agent")

    def _set_att_blend_dist(self, params):
        """ Used to set up blending distribution.

        Parameters
        ----------
        params : list/ndarray
            List of distribution parameters: [mean, std]

        """
        lower, upper = 0, 1
        self.mu = params[0]
        self.sigma = params[1]
        self.blendDist = stats.truncnorm((lower - self.mu) / self.sigma, (upper - self.mu) / self.sigma, loc=self.mu,
                                         scale=self.sigma)

    def _get_blend(self):
        """ Used to sample new weights at each iteration from the currently
        defined distribution ( self.blendDist - gaussian dist defined using mean and std.)

        """
        return self.blendDist.rvs(size=1)

    def is_at_pos(self, pos):
        [dest_x, dest_y, dest_z] = pos
        [x, y, z, _, _, _, _, _, _, _, _, _] = self.get_state(self.quad_identifier)
        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z
        total_distance_to_goal = abs(x_error) + abs(y_error) + abs(z_error)  # L1 norm

        isAt = True if total_distance_to_goal < self.goal_safety_margin else False

        return isAt

    def get_distance_to_opt(self):
        # Get closest point on the linspace between waypoints
        [x, y, z, _, _, _, _, _, _, _, _, _] = self.get_state(self.quad_identifier)

        current_pos = [x, y, z]
        distances = []
        points = []
        for i in range(len(self.safe_bound)):
            safebound_center = np.array([self.safe_bound[i][0], self.safe_bound[i][1], self.safe_bound[i][2]])
            dist = np.linalg.norm(current_pos - safebound_center)
            distances.append(dist)
            points.append(safebound_center)

        index = np.argmin(distances)
        return distances[index]

    def check_safety_bound(self):
        self.current_distance_to_opt = self.get_distance_to_opt()

        if self.current_distance_to_opt > self.safety_margin:
            self.outsideBounds = True
        else:
            self.outsideBounds = False

        return self.outsideBounds

    def get_reward(self):

        if self.outsideBounds:
            # left safety region give negative reward
            reward = -1
        else:
            reward = 0
        # if self.total_steps > 5000:
        #     reward = -0.1

        return reward

    def update_target(self, target, new_safety_bound):
        self.target = target
        self.safe_bound = new_safety_bound
        self.current_distance_to_opt = self.get_distance_to_opt()

    def _set_quadcopter_motor_fault(self):
        self.set_motor_faults(self.quad_identifier, self.motor_faults)
        return

    def _clear_quadcopter_motor_fault(self):
        self.set_motor_faults(self.quad_identifier, [0, 0, 0, 0])
        return

    def _set_steady_wind_fault(self):
        rng = default_rng()
        self.set_normal_wind(self.normal_wind)
        if self.total_steps % 20 == 0:
            randWind = rng.normal(0, 0.5, size=3)
            self.setWind(randWind)
        return

    def _clear_steady_wind_fault(self):
        self.set_normal_wind([0, 0, 0])
        self.setWind([0, 0, 0])
        return

    # Step control
    def update(self, return_weight=False, weight_clip=None):
        """ The main functions that step the simulation forward and set the commands for the
        quadcopter. Can be roughly broken down as follows:

        Step 1: get current state update of quadcopter
        Step 2: Add noise to the state if noise is enabled as faultmode
        Step 3: get Position error and use Linear PID to get the attitude reference
        Step 4: use those to calculate the attitude error
        Step 5: Get the suggested actions from all of the attitude PID controllers configured.
        Step 6: Depending on the high-level control architecture selected - get a blending weight
        Step 7: Calculate the four motor commands based on weighted actions of all controllers.
        Step 8: Apply those to the quadcopter and get new observation of quadcopter states.
        """
        rng = default_rng()
        self.total_steps += 1

        if self.check_safety_bound():
            self.total_steps_out_of_bounds = self.total_steps_out_of_bounds + 1

        [dest_x, dest_y, dest_z] = self.target
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state(
            self.quad_identifier)

        self.trajectory.append([x, y, z])

        # Noise
        pos, att = self.add_noise([x, y, z], [theta, phi, gamma])
        x, y, z = pos
        theta, phi, gamma = att

        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z

        self.xi_term += self.LINEAR_I[0] * x_error
        self.yi_term += self.LINEAR_I[1] * y_error
        self.zi_term += self.LINEAR_I[2] * z_error
        dest_x_dot = self.LINEAR_P[0] * x_error + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1] * y_error + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2] * z_error + self.LINEAR_D[2]*(-z_dot) + self.zi_term

        throttle = np.clip(dest_z_dot, self.Z_LIMITS[0], self.Z_LIMITS[1])

        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))

        # Get required attitude states
        dest_gamma = self.yaw_target
        dest_theta = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1])
        dest_phi = np.clip(dest_phi, self.TILT_LIMITS[0], self.TILT_LIMITS[1])

        theta_error = dest_theta - theta
        phi_error = dest_phi - phi
        gamma_error = dest_gamma - gamma
        gamma_dot_error = (self.YAW_RATE_SCALER * wrap_angle(gamma_error)) - gamma_dot

        self.trackingErrors["Pos_err"] += abs(round(x_error, 2)) + abs(round(y_error, 2)) + abs(round(z_error, 2))
        self.trackingErrors["Att_err"] += abs(round(phi_error, 2)) + abs(round(theta_error, 2)) + \
                                          abs(round(gamma_error, 2))

        # -----------------------------------------------------------------------
        #  GET DIFFERENT CONTROL ARCHITECTURE OUTPUTS - only apply the selected
        # -----------------------------------------------------------------------
        # Controller 1
        self.thetai_term += self.ANGULAR_I[0] * theta_error
        self.phii_term += self.ANGULAR_I[1] * phi_error
        self.gammai_term += self.ANGULAR_I[2] * gamma_dot_error

        x_val = self.ANGULAR_P[0] * theta_error + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1] * phi_error + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2] * gamma_dot_error + self.gammai_term
        z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])

        # Controller 2
        self.thetai_term2 += self.ANGULAR_I2[0] * theta_error
        self.phii_term2 += self.ANGULAR_I2[1] * phi_error
        self.gammai_term2 += self.ANGULAR_I2[2] * gamma_dot_error

        x_val2 = self.ANGULAR_P2[0] * theta_error + self.ANGULAR_D2[0] * (-theta_dot) + self.thetai_term2
        y_val2 = self.ANGULAR_P2[1] * phi_error + self.ANGULAR_D2[1] * (-phi_dot) + self.phii_term2
        z_val2 = self.ANGULAR_P2[2] * gamma_dot_error + self.gammai_term2
        z_val2 = np.clip(z_val2, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])

        # calculate motor commands depending on controller selection
        if self.controller_type == "C1":
            dist_blend_weight = 0
            m1 = throttle + x_val + z_val
            m2 = throttle + y_val - z_val
            m3 = throttle - x_val + z_val
            m4 = throttle - y_val - z_val
        elif self.controller_type == "C2":
            dist_blend_weight = 1
            m1 = throttle + x_val2 + z_val2
            m2 = throttle + y_val2 - z_val2
            m3 = throttle - x_val2 + z_val2
            m4 = throttle - y_val2 - z_val2
        # Blended controller
        elif self.controller_type == "Agent":
            # USE THE SAME DISTRIBUTION FOR ROLL AND PITCH
            # THESE DISTRIBUTIONS ARE UPDATED BY THE AGENT USING set_action()

            dist_blend_weight = self._get_blend()[0]
            if weight_clip is not None:
                dist_blend_weight = np.clip(dist_blend_weight, weight_clip[0], weight_clip[1])
            blend_weight = [dist_blend_weight, dist_blend_weight, 0]  # Only one blend weight. Equivalent to 2 actions.

            x_val_blend = x_val2 * blend_weight[0] + x_val * (1 - blend_weight[0])
            y_val_blend = y_val2 * blend_weight[1] + y_val * (1 - blend_weight[1])
            z_val_blend = z_val2 * blend_weight[2] + z_val * (1 - blend_weight[2])

            m1 = throttle + x_val_blend + z_val_blend
            m2 = throttle + y_val_blend - z_val_blend
            m3 = throttle - x_val_blend + z_val_blend
            m4 = throttle - y_val_blend - z_val_blend
        elif self.controller_type == "Uniform":
            # USE THE SAME DISTRIBUTION FOR ROLL AND PITCH
            # THIS IS A NORMAL DISTRIBUTION

            dist_blend_weight = rng.uniform(0, 1)
            if weight_clip is not None:
                dist_blend_weight = np.clip(dist_blend_weight, weight_clip[0], weight_clip[1])
            blend_weight = [dist_blend_weight, dist_blend_weight, 0]

            x_val_blend = x_val2 * blend_weight[0] + x_val * (1 - blend_weight[0])
            y_val_blend = y_val2 * blend_weight[1] + y_val * (1 - blend_weight[1])
            z_val_blend = z_val2 * blend_weight[2] + z_val * (1 - blend_weight[2])

            m1 = throttle + x_val_blend + z_val_blend
            m2 = throttle + y_val_blend - z_val_blend
            m3 = throttle - x_val_blend + z_val_blend
            m4 = throttle - y_val_blend - z_val_blend
        else:
            print("No control architecture selected")
            m1 = throttle + x_val + z_val
            m2 = throttle + y_val - z_val
            m3 = throttle - x_val + z_val
            m4 = throttle - y_val - z_val

        M = np.clip([m1, m2, m3, m4], self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])

        # Check for rotor fault to inject to quad
        if self.FaultMode == "Rotor":
            if self.fault_time[0] <= self.total_steps <= self.fault_time[1]:
                self._set_quadcopter_motor_fault()
                self.loe_log.append(True)
            else:
                self.loe_log.append(False)
                self._clear_quadcopter_motor_fault()

        if self.FaultMode == "Wind":
            if self.fault_time[0] <= self.total_steps <= self.fault_time[1]:
                self._set_steady_wind_fault()
            else:
                self._clear_steady_wind_fault()

        if self.FaultMode == 'Intermittent_Rotor':
            self.loe_log.append(self.intermittent_rotor_on)
            if self.fault_time[0] <= self.total_steps <= self.fault_time[1]:
                if (self.total_steps - self.fault_time[0]) % self.rotor_switching_period == 0:
                    self.intermittent_rotor_on = not self.intermittent_rotor_on
            else:
                self.intermittent_rotor_on = False
            if self.intermittent_rotor_on:
                self._set_quadcopter_motor_fault()
            else:
                self._clear_quadcopter_motor_fault()

        self.actuate_motors(self.quad_identifier, M)
        self.motor_commands = M

        # Step the quad to the next state with the new commands
        self.step_quad(0.01)   # Control period of 0.01s
        self.generate_noise()
        new_obs = self.get_observations()

        # Return logic
        if return_weight:
            return new_obs, dist_blend_weight
        else:
            return new_obs

    def set_action(self, action, weight_clip=None):
        """ Gives the agent a way to influence the main control
        loop by changing the Blending Distribution to use

        Parameters
        ----------
        action : list/ndarray
            List of distribution parameters: [mean, std]
        weight_clip : list
        """
        self._set_att_blend_dist(action)

        # Steps simulation forward
        obs, blendweight = self.update(return_weight=True, weight_clip=weight_clip)
        return obs, blendweight

    def generate_noise(self):
        rng = default_rng()
        if self.fault_time[0] <= self.total_steps <= self.fault_time[1]:
            if self.FaultMode == "PosNoise":
                self.x_noise = rng.uniform(-self.noiseMag, self.noiseMag)
                self.y_noise = rng.uniform(-self.noiseMag, self.noiseMag)
                self.z_noise = rng.uniform(-self.noiseMag, self.noiseMag)
            if self.FaultMode == "AttNoise":
                self.theta_noise = rng.uniform(-self.attNoiseMag, self.attNoiseMag)
                self.phi_noise = rng.uniform(-self.attNoiseMag, self.attNoiseMag)
                self.gamma_noise = rng.uniform(-self.attNoiseMag, self.attNoiseMag)
        else:
            self.x_noise = 0.0
            self.y_noise = 0.0
            self.z_noise = 0.0
            self.theta_noise = 0.0
            self.phi_noise = 0.0
            self.gamma_noise = 0.0

    def add_noise(self, position=[0, 0, 0], attitude=[0, 0, 0]):

        if self.FaultMode == "PosNoise":
            position[0] = position[0] + self.x_noise
            position[1] = position[1] + self.y_noise
            position[2] = position[2] + self.z_noise
        if self.FaultMode == "AttNoise":
            attitude[0] = attitude[0] + self.theta_noise
            attitude[1] = attitude[1] + self.phi_noise
            attitude[2] = attitude[2] + self.gamma_noise

        return position, attitude

    # Setters
    def set_position_sensor_noise(self, noise):
        self.noiseMag = noise
        self.generate_noise()

    def set_attitude_sensor_noise(self, noise):
        self.attNoiseMag = noise
        self.generate_noise()

    def set_motor_fault(self, fault):
        self.motor_faults = fault  # should be 0-1 value for each motor

    def set_intermittent_motor_fault(self, fault, period):
        self.motor_faults = fault  # should be 0-1 value for each motor
        self.rotor_switching_period = period

    def set_steady_wind_magnitude(self, fault):
        self.normal_wind = fault  # should be a 3dim vector with x,y,z wind magnitudes and directions.

    def set_controller(self, ctrl):
        if ctrl == "C1":
            self.controller_type = "C1"
        elif ctrl == "C2":
            self.controller_type = "C2"
        elif ctrl == "Agent":
            self.controller_type = "Agent"
        elif ctrl == "Uniform":
            self.controller_type = "Uniform"
        else:
            print("Controller not available")

    def set_fault_time(self, low, high):
        self.startfault = low
        self.endfault = high
        self.fault_time = [self.startfault, self.endfault]

    def set_fault_mode(self, mode):
        self.FaultMode = mode

    def set_normal_wind(self, winds):
        self.setNormWind(winds)

    # Getters
    def get_tracking_errors(self):
        err_array = []
        print(self.trackingErrors)
        for key, value in self.trackingErrors.items():
            err_array.append(value/self.total_steps)
        return err_array

    def get_trajectory(self):
        return self.trajectory

    def get_total_steps(self):
        return self.total_steps

    def get_observations(self):
        """ Returns observation ready for agent. Not a full state representation.
        """
        [x, y, z, _, _, _, theta, phi, gamma, _, _, _] = self.get_state(self.quad_identifier)
        pos, att = self.add_noise([x, y, z], [theta, phi, gamma])
        x, y, z = pos
        theta, phi, gamma = att

        [dest_x, dest_y, dest_z] = self.target
        obs = [x, y, z, theta, phi, gamma, dest_x, dest_y, dest_z]

        return obs

    def get_full_observation(self):
        state = self.get_state(self.quad_identifier)
        control = self.motor_commands

        return state, control


################### Helper Functions  #################################
def wrap_angle(val):
    return (val + np.pi) % (2 * np.pi) - np.pi
