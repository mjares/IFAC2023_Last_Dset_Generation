import scipy.integrate
import time
import datetime
import signal
import threading
import scipy.stats as stats
import numpy as np
from numpy.random import default_rng
import math
from scipy import signal


lower, upper = 0, 1
mu = 0
sigma = 1
randDist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu,
                                 scale=sigma)


class Propeller:
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0  #RPM
        self.thrust = 0
        self.fault_mag = 0

    def set_speed(self,speed):
        self.speed = speed
        if self.fault_mag > 0:
            self.speed = self.speed * (1 - self.fault_mag)

        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia, 3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)

        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust*0.101972

    def get_speed(self):
        return self.speed

    def set_fault(self,fault):
        self.fault_mag = fault


class Quadcopter:
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self, quads, gravity=9.81, b=0.0245):
        self.quads = quads
        self.g = gravity
        self.b = b
        self.wind = True
        self.windMag = 0
        self.thread_object = None
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')
        self.time = datetime.datetime.now()
        self.stepNum = 0
        self.airspeed = 15
        # self.randWind = self.generate_wind_turbulence(5)  # Default values
        self.randWind = [0, 0, 0]  # Only for wind conditions
        self.XWind = 0
        self.YWind = 0
        self.ZWind = 0

        for key in self.quads:
            self.quads[key]['state'] = np.zeros(12)
            self.quads[key]['state'][0:3] = self.quads[key]['position']
            self.quads[key]['state'][6:9] = self.quads[key]['orientation']
            self.quads[key]['m1'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            self.quads[key]['m2'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            self.quads[key]['m3'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            self.quads[key]['m4'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            # From Quadrotor Dynamics and Control by Randal Beard
            ixx = ((2*self.quads[key]['weight'] * self.quads[key]['r']**2) / 5) + \
                  (2 * self.quads[key]['weight'] * self.quads[key]['L']**2)
            iyy = ixx
            izz = ((2 * self.quads[key]['weight'] * self.quads[key]['r']**2) / 5) + \
                  (4 * self.quads[key]['weight'] * self.quads[key]['L']**2)
            self.quads[key]['I'] = np.array([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
            self.quads[key]['invI'] = np.linalg.inv(self.quads[key]['I'])
        self.run = True

    def rotation_matrix(self, angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def setWind(self, wind_vec):
        self.randWind = wind_vec

    def setNormalWind(self, winds):
        self.XWind = winds[0]
        self.YWind = winds[1]
        self.ZWind = winds[2]

    def state_dot(self, time, state, key):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.quads[key]['state'][3]
        state_dot[1] = self.quads[key]['state'][4]
        state_dot[2] = self.quads[key]['state'][5]
        # The acceleration
        height = self.quads[key]['state'][2]
        #
        F_d = np.array([ 0, 0, 0])
        #
        air_density = 1.225  # kg/m^3
        C_d = 1
        cube_width = 0.1  # 10cm x 10cm cube as shape model of quadcopter
        A_yz = cube_width*cube_width
        A_xz = cube_width*cube_width
        A_xy = cube_width*cube_width

        A = [A_yz, A_xz, A_xy]  # cross sectional area in each axis perpendicular to velocity axis

        #if wind is active the velocity in each axis is subject to wind
        nomX = self.XWind
        nomY = self.YWind
        nomZ = self.ZWind

        if self.stepNum > 19500:
            self.stepNum = 0
        # Only for wind conditions
        randX = self.randWind[0]
        randY = self.randWind[1]
        randZ = self.randWind[2]
        # For rest of scenarios
        # randX = self.randWind[0][self.stepNum]
        # randY = self.randWind[1][self.stepNum]
        # randZ = self.randWind[2][self.stepNum]

        wind_velocity_vector = [nomX + randX, nomY + randY, nomZ + randZ]  # wind velocity in each axis

        wind_vel_inertial_frame = np.dot(self.rotation_matrix(self.quads[key]['state'][6:9]), wind_velocity_vector)
        V_b = [state[0], state[1], state[2]]
        V_a = wind_vel_inertial_frame - V_b

        DragVector = [
            A[0] * (V_a[0] * abs(V_a[0])),
            A[1] * (V_a[1] * abs(V_a[1])),
            A[2] * (V_a[2] * abs(V_a[2]))
        ]

        F_d = [i * (0.5 * air_density * C_d) for i in DragVector]
        # form drag is a -0.5 and wind seems to be a +0.5. why??
        # the velocity is subtracted from wind ?

        #************* This line needs cleaning
        x_dotdot = np.array([0, 0, -self.quads[key]['weight']*self.g]) + np.dot(self.rotation_matrix(self.quads[key]['state'][6:9]),
                            np.array([0, 0, (self.quads[key]['m1'].thrust + self.quads[key]['m2'].thrust +
                            self.quads[key]['m3'].thrust + self.quads[key]['m4'].thrust)])) / \
                            self.quads[key]['weight'] + F_d

        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.quads[key]['state'][9]
        state_dot[7] = self.quads[key]['state'][10]
        state_dot[8] = self.quads[key]['state'][11]
        # The angular accelerations
        omega = self.quads[key]['state'][9:12]
        tau = np.array([self.quads[key]['L']*(self.quads[key]['m1'].thrust-self.quads[key]['m3'].thrust),
                        self.quads[key]['L']*(self.quads[key]['m2'].thrust-self.quads[key]['m4'].thrust),
                        self.b*(self.quads[key]['m1'].thrust-self.quads[key]['m2'].thrust +
                                self.quads[key]['m3'].thrust-self.quads[key]['m4'].thrust)])
        omega_dot = np.dot(self.quads[key]['invI'], (tau - np.cross(omega, np.dot(self.quads[key]['I'], omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def update(self, dt):
        self.stepNum += 1
        for key in self.quads:
            self.ode.set_initial_value(self.quads[key]['state'], 0).set_f_params(key)
            self.quads[key]['state'] = self.ode.integrate(self.ode.t + dt)
            self.quads[key]['state'][6:9] = self.wrap_angle(self.quads[key]['state'][6:9])
            self.quads[key]['state'][2] = max(0, self.quads[key]['state'][2])

    def generate_wind_turbulence(self, h):
        rng = default_rng()
        height = float(h) * 3.28084
        airspeed = float(self.airspeed) * 3.28084

        mean = 0
        std = 1
        # create a sequence of 20000 equally spaced numeric values from 0 - 10
        t_p = np.linspace(0, 10, 20000)
        num_samples = 20000
        t_p = np.linspace(0, 10, 20000)

        # the random number seed used same as from SIMULINK blockset
        np.random.seed(23341)
        samples1 = 10 * np.random.normal(mean, std, size=num_samples)

        np.random.seed(23342)
        samples2 = 10 * np.random.normal(mean, std, size=num_samples)

        np.random.seed(23343)
        samples3 = 10 * np.random.normal(mean, std, size=num_samples)

        tf_u = u_transfer_function(height, airspeed)
        tf_v = v_transfer_function(height, airspeed)
        tf_w = w_transfer_function(height, airspeed)

        tout1, y1, x1 = signal.lsim(tf_u, samples1, t_p)
        # covert obtained values to meters/second
        y1_f = [i * 0.305 for i in y1]
        tout2, y2, x2 = signal.lsim(tf_v, samples2, t_p)
        y2_f = [i * 0.305 for i in y2]
        tout3, y3, x3 = signal.lsim(tf_w, samples3, t_p)
        y3_f = [i * 0.305 for i in y3]

        return [y1_f, y2_f, y3_f]

    def set_motor_speeds(self, quad_name, speeds):
        self.quads[quad_name]['m1'].set_speed(speeds[0])
        self.quads[quad_name]['m2'].set_speed(speeds[1])
        self.quads[quad_name]['m3'].set_speed(speeds[2])
        self.quads[quad_name]['m4'].set_speed(speeds[3])

    def get_motor_speeds(self,quad_name):
        return [self.quads[quad_name]['m1'].get_speed(), self.quads[quad_name]['m2'].get_speed(),
                self.quads[quad_name]['m3'].get_speed(), self.quads[quad_name]['m4'].get_speed()]

    def get_motor_speeds_rpm(self,quad_name):
        return [self.quads[quad_name]['m1'].get_speed(), self.quads[quad_name]['m2'].get_speed(),
                self.quads[quad_name]['m3'].get_speed(), self.quads[quad_name]['m4'].get_speed()]

    def get_position(self,quad_name):
        return self.quads[quad_name]['state'][0:3]

    def get_linear_rate(self,quad_name):
        return self.quads[quad_name]['state'][3:6]

    def get_orientation(self,quad_name):
        return self.quads[quad_name]['state'][6:9]

    def get_angular_rate(self,quad_name):
        return self.quads[quad_name]['state'][9:12]

    def get_state(self,quad_name):
        return self.quads[quad_name]['state']

    def set_position(self,quad_name,position):
        self.quads[quad_name]['state'][0:3] = position

    def set_orientation(self,quad_name,orientation):
        self.quads[quad_name]['state'][6:9] = orientation

    def get_time(self):
        return self.time

    def thread_run(self, dt, time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while self.run:
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self.time

    def stepQuad(self, dt=0.05):
        self.update(dt)
        return

    def set_motor_faults(self, quad_name, faults):
        f1 = faults[0]
        f2 = faults[1]
        f3 = faults[2]
        f4 = faults[3]
        self.quads[quad_name]['m1'].set_fault(f1)
        self.quads[quad_name]['m2'].set_fault(f2)
        self.quads[quad_name]['m3'].set_fault(f3)
        self.quads[quad_name]['m4'].set_fault(f4)

    def start_thread(self, dt=0.002, time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run, args=(dt, time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False


# Low altitude Model
# transfer function for along-wind
def u_transfer_function(height, airspeed):
    # turbulence level defines value of wind speed in knots at 20 feet
    # turbulence_level = 15 * 0.514444 # convert speed from knots to meters per second
    turbulence_level = 15
    length_u = height / ((0.177 + 0.00823 * height)**0.2)
    sigma_w = 0.1 * turbulence_level
    sigma_u = sigma_w / ((0.177 + 0.000823 * height)**0.4)
    num_u = [sigma_u * (math.sqrt((2 * length_u) / (math.pi * airspeed))) * airspeed]
    den_u = [length_u, airspeed]
    H_u = signal.TransferFunction(num_u, den_u)
    return H_u


# transfer function for cross-wind
def v_transfer_function(height, airspeed):
    # turbulence level defines value of wind speed in knots at 20 feet
    # turbulence_level = 15 * 0.514444 # convert speed from knots to meters per second
    turbulence_level = 15
    length_v = height / ((0.177 + 0.00823 * height)**0.2)
    sigma_w = 0.1 * turbulence_level
    sigma_v = sigma_w / ((0.177 + 0.000823 * height)**0.4)
    b = sigma_v * (math.sqrt(length_v / (math.pi * airspeed)))
    Lv_V = length_v / airspeed
    num_v = [(math.sqrt(3) * Lv_V * b), b]
    den_v = [(Lv_V ** 2), 2 * Lv_V, 1]
    H_v = signal.TransferFunction(num_v, den_v)
    return H_v


# transfer function for vertical-wind
def w_transfer_function(height, airspeed):
    # turbulence level defines value of wind speed in knots at 20 feet
    # turbulence_level = 15 * 0.514444 # convert speed from knots to meters per second
    turbulence_level = 15
    length_w = height
    sigma_w = 0.1 * turbulence_level
    c = sigma_w * (math.sqrt(length_w / (math.pi * airspeed)))
    Lw_V = length_w / airspeed
    num_w = [(math.sqrt(3) * Lw_V * c), c]
    den_w = [(Lw_V ** 2), 2 * Lw_V, 1]
    H_v = signal.TransferFunction(num_w, den_w)
    return H_v

