import numpy as np

import control.utils as utils
import shared.resource.car_info as car_info


# Constants
DELTA = 0.001
L = car_info.L
MAX_WHEEL_ANGLE = 0.5 # rad
MAX_SPEED = 10.0/3.6 # m/s



class Controler:
    def __init__(self, X=[0,0,0], ed=0.0, vd=MAX_SPEED):
        """
        This class contains useful informations about the car

        Parameters
        ----------
        X: list
            State of the vehicle (x, y, theta)
        ed: float
            Initial desired error (for delta)
        vd: float
            Initial desired speed
        """
        self.map_to_follow = np.array([[]])
        self.is_stoping = False

        self.simulated_state = X.copy()
        self.observed_state = X.copy()
        self.front_axle_state = X.copy()
        self.Te = 0.0

        self.wheel_angle = 0.0
        self.desired_lateral_error = ed
        self.old_wheel_angle = self.wheel_angle

        self.lateral_error = 0.0
        self.angle_error = 0.0
        self.curvature = 0.0
        
        self.acceleration = 0.0
        self.observed_speed = 0.0
        self.simulated_speed = 0.0
        self.desired_speed = min(vd, MAX_SPEED)
        self.epsilon = 0.0
        self.integral = 0.0

        self.closest_segment = np.array([0,0])
        self.closest_segment2 = np.array([0,0])

        self.gain_proportionnel = 4.0
        self.gain_integrale = 0.4
        self.integral_max = 5.0

        self.gain_ecart = 0.1
        self.gain_pente = 0.5
        self.gain_courbure = 0.0

        self.a_lat_max = 0.8
        self.front_axle_distance = 0.0
        self.delta_speed_factor = 0.5

        self.delta_ecart = 0
        self.delta_pente = 0
        self.delta_courbure = 0


    def set_map_to_follow(self, m):
        self.map_to_follow = m


    def set_observed_state(self, x, y, theta):
        self.observed_state = [x,y,theta]


    # State space
    def f(self, X, delta, v, L):
        x_dot = v * np.cos(delta) * np.cos(X[2])
        y_dot = v * np.cos(delta) * np.sin(X[2])
        theta_dot = v * np.sin(delta) / L
        return np.array([x_dot, y_dot, theta_dot])


    def one_step(self, simulation=False):
        """
        Make one step of the simulation (lateral and longitudinal control)

        Returns
        -------

        self.wheel_angle: float
            The output angle of the steering wheel (rad)
        self.acceleration: float
            The output acceleration of the car
        """

        if self.is_stoping:
            self.desired_speed = 0.0

        if self.map_to_follow.size == 0:
            return 0, 0

        if simulation:
            state = self.simulated_state
            speed = self.simulated_speed
        else:
            state = self.observed_state
            speed = self.observed_speed

            # Computing lateral error, angle error, and curvature 
        self.front_axle_state = state.copy()
        self.front_axle_state[0] += np.cos(state[2]) * (car_info.EMPATTEMENT + self.front_axle_distance)
        self.front_axle_state[1] += np.sin(state[2]) * (car_info.EMPATTEMENT + self.front_axle_distance)

        e_lat_front_axle, e_rot_front_axle, self.closest_segment, self.closest_segment2 = utils.closest_segment(self.front_axle_state, self.map_to_follow, self.closest_segment)

        self.lateral_error = e_lat_front_axle
        self.angle_error = utils.sub_mod_pi2(e_rot_front_axle, self.front_axle_state[2])
        self.curvature = utils.estimate_curvature(self.map_to_follow, self.closest_segment, self.closest_segment2, self.front_axle_state[:2])

            # Lateral control law (PD+curvature: equation 8.17, p.100)
        self.old_wheel_angle = self.wheel_angle
        e = (self.desired_lateral_error-self.lateral_error)

        delta_speed = speed * self.delta_speed_factor # Speed used for delta calculation: the lower the more reactive the steering is
        self.delta_ecart = self.gain_ecart * L / (delta_speed**2+DELTA) * e             # Lateral error component
        self.delta_pente = self.gain_pente * L / (delta_speed+DELTA) * self.angle_error # Angle error component
        self.delta_courbure = -self.gain_courbure * L * self.curvature                  # Curvature component
        self.wheel_angle = self.delta_ecart + self.delta_pente + self.delta_courbure
        
        self.wheel_angle = np.clip(self.wheel_angle, -MAX_WHEEL_ANGLE, MAX_WHEEL_ANGLE) # Steering angle saturation
        old_delta_coef = np.clip(1.0 - self.Te, 0.0, 1.0)
        self.wheel_angle = old_delta_coef * self.old_wheel_angle + (1.0-old_delta_coef) * self.wheel_angle

            # Longitudinal control law (PI: p.104)
        # Reducing desired speed with curvature
        capped_speed = np.sqrt(self.a_lat_max / max(abs(self.curvature), 1e-6))
        self.desired_speed = 0.0 if self.is_stoping else min(capped_speed, MAX_SPEED)
        # Computing acceleration
        epsilon = self.desired_speed - speed
        self.integral += epsilon * self.Te
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        self.acceleration = self.gain_proportionnel*epsilon + self.gain_integrale*self.integral
        speed += self.acceleration * self.Te

            # Simulation
        self.simulated_state = state + self.Te * self.f(state, self.wheel_angle, speed, L)
        self.simulated_speed = speed
        if simulation:
            self.observed_state = self.simulated_state
            self.observed_speed = self.simulated_speed

        return self.wheel_angle, self.acceleration
