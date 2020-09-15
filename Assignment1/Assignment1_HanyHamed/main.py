import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, sqrt, asin, acos

OM = 3
A = 1
theta_0 = 0.2
dt = 0.001

class TrapezoidalMotion_mod():
    def __init__(self, max_velocity, max_acceleration_tangential, max_acceleration_normal, initial_pos=0, initial_vel=0, initial_curvature=0, dt=0.001):
        self.MAX_ACCELERATION_TANGENTIAL = max_acceleration_tangential
        self.MAX_ACCELERATION_NORMAL = max_acceleration_normal
        self.MAX_ACCELERATION = sqrt(self.MAX_ACCELERATION_NORMAL**2 + self.MAX_ACCELERATION_TANGENTIAL**2)
        self.MAX_VELOCITY = max_velocity

        self.MAX_VELOCITY_var = max_velocity
        
        self.old_pos = initial_pos
        self.old_pos_ref = initial_pos
        self.old_vel = initial_vel
        self.pos = initial_pos
        self.vel = initial_vel
        self.acc = 0

        self.total_displacement = self.old_pos_ref - self.old_pos

        self.t_a = 0
        self.T = 0
        self.t_f = 0
        self.t_i = 0
        self.delta_time = dt
        self.current_time = 0

        self.curvature = self._calc_curvature()


    def _calc_limited_vel(self):
        velocity = sqrt(self.MAX_ACCELERATION_NORMAL/self._calc_curvature())
        return velocity

    def _calc_max_vel(self):
        # print(self._calc_limited_vel(), self.MAX_VELOCITY)
        return min(self.MAX_VELOCITY, self._calc_limited_vel())


    def _calc_curvature(self):
        x = self.current_time
        dot_f = A*cos(OM*x+theta_0)*OM
        ddot_f = -A*sin(OM*x+theta_0)*OM*OM
        k = abs(ddot_f/(pow((1+dot_f*dot_f),(3/2))))
        return k

    def _calc_normal_acceleration(self):
        return self.vel**2*self._calc_curvature()

    def _calc_tangential_acceleration(self):
        return (self.vel-self.old_vel)/self.delta_time

    def update(self, pos_ref, time):
        self.current_time = time
        self.old_vel = self.vel
        if(self.old_pos_ref != pos_ref):
            self.old_pos = self.pos
            self.old_pos_ref = pos_ref
            self.t_i = time

            self.MAX_VELOCITY_var = self._calc_max_vel()
            self.t_a = (self.MAX_VELOCITY_var - abs(self.vel))/self.MAX_ACCELERATION
            self.t_d = (self.vel)/self.MAX_ACCELERATION
            self.total_displacement = abs(pos_ref - self.pos)
            # self.T = t_a + t_d
            self.T = (self.total_displacement*self.MAX_ACCELERATION+self.MAX_VELOCITY_var**2)/(self.MAX_ACCELERATION*self.MAX_VELOCITY_var)
            self.t_f = self.T + time

        self.calculateTrapezoidalProfile()


    def calculateTrapezoidalProfile(self):
        # if(self.t_i <= self.current_time and self.current_time <= self.t_i+self.t_a):
        if(self.delta_time <= self.t_a):
            # print("+ Acc")
            self.pos = self.pos + self.vel*self.delta_time
            # self.pos = self.old_pos + 1/2*self.MAX_ACCELERATION*((self.current_time - self.t_i)**2)
            self.vel = self.vel + self.MAX_ACCELERATION*self.delta_time
            # print(self.vel)
            self.acc = self.MAX_ACCELERATION

        # elif(self.t_i + self.t_a < self.current_time and self.current_time <= self.t_f - self.t_a):
        elif(self.delta_time > self.t_a and self.delta_time < self.T - self.t_a - self.t_d):
            # print("0 Acc")
            self.pos = self.pos + self.vel*self.delta_time
            # self.pos = self.old_pos + self.MAX_ACCELERATION*((self.current_time-self.t_i-(self.t_a/2))**2)
            self.vel = self.MAX_VELOCITY_var
            self.acc = 0
            
        # elif(self.t_f - self.t_a < self.current_time and self.current_time <= self.t_f):
        elif(self.delta_time >= self.T - self.t_a - self.t_d and self.delta_time <= self.t_d):
            # print("- Acc")
            self.pos = self.pos + self.vel*self.delta_time
            # self.pos = self.old_pos_ref - (0.5*self.MAX_ACCELERATION*((self.t_f-self.current_time-self.t_i)**2))
            self.vel = self.vel - self.MAX_ACCELERATION*self.delta_time
            self.acc = -self.MAX_ACCELERATION

def plotter(plot_data):
    # plt.subplot()
    # Add the units to the axis

    fig = plt.figure()
    ax = plt.subplot(511)
    ax.plot(plot_data["pos_x_setpoint"], plot_data["pos_y_setpoint"], label="Setpoint-Trajectory")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.)
    # ax.set(title="Position", xlabel='x(m)', ylabel='y(m)')
    ax.set_title("Position",fontsize=12)
    ax.set_xlabel("x(m)",fontsize=7)
    ax.set_ylabel('y(m)', fontsize=7)


    ax = plt.subplot(512)
    ax.plot(plot_data["time"], plot_data["curvature"], "--", label="Curvature")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.)
    ax.set_title("Curvature vs. Time",fontsize=12)
    ax.set_xlabel("t(s)",fontsize=7)
    ax.set_ylabel('k', fontsize=7)
    
    ax = plt.subplot(513)
    ax.plot(plot_data["pos_x_setpoint"], plot_data["pos_y_setpoint"], label="Setpoint-Trajectory")
    ax.plot(plot_data["time"], plot_data["pos_y_real"], label="Real-Trajectory")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.)
    ax.set_title("Position",fontsize=12)
    ax.set_xlabel("x(m)",fontsize=7)
    ax.set_ylabel('y(m)', fontsize=7) 

    ax = plt.subplot(514)
    ax.plot(plot_data["time"], plot_data["velocity"], label="Velocity")
    ax.plot(plot_data["time"], plot_data["max_velocity"], "--", label="Max Velocity Planning")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.)
    ax.set_title("Velocity vs Time",fontsize=12)
    ax.set_xlabel("t(s)",fontsize=7)
    ax.set_ylabel('v(m/s)', fontsize=7)   

    ax = plt.subplot(515)
    ax.plot(plot_data["time"], plot_data["acceleration"], label="Total Acceleration")
    ax.plot(plot_data["time"], plot_data["acceleration_tangential"], "--", label="Tangential Acceleration")
    ax.plot(plot_data["time"], plot_data["acceleration_normal"], "--", label="Normal Acceleration")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.)
    ax.set_title("Acceleration vs Time",fontsize=12)
    ax.set_xlabel("t(s)",fontsize=7)
    ax.set_ylabel('a\n(m/s^2)', fontsize=7)

    # Add the units to the axis
    fig.canvas.set_window_title('Plotting')
    plt.tight_layout()
    fig.tight_layout()

    plt.show()

def motion_equation(t):
    x = t
    y = A*sin(OM*x+theta_0)
    return x,y

def main(time):
    total_steps = int(time/dt)

    MAX_VELOCITY = 1.5
    MAX_ACCELERATION_TANGENTIAL = 10
    MAX_ACCELERATION_NORMAL = 6
    MAX_ACCELERATION = sqrt(MAX_ACCELERATION_TANGENTIAL**2 + MAX_ACCELERATION_NORMAL**2)
    
    traj_planner = TrapezoidalMotion_mod(max_velocity=MAX_VELOCITY, max_acceleration_normal=MAX_ACCELERATION_NORMAL, max_acceleration_tangential=MAX_ACCELERATION_TANGENTIAL, initial_pos=motion_equation(0)[1], initial_vel=0, dt=dt)


    plot_data = {"time": [0],
                 "pos_x_setpoint": [0], "pos_y_setpoint": [motion_equation(0)[1]], "curvature": [traj_planner._calc_curvature()],
                 "pos_y_real": [0],
                 "velocity": [0], "velocity_x": [0], "velocity_y": [0], "max_velocity": [0],
                 "acceleration": [0], "acceleration_tangential": [0], "acceleration_normal": [0]}

    for s in range(1, total_steps):
        pos = motion_equation(s*dt)
        traj_planner.update(pos[1], s*dt)
        curvature = traj_planner._calc_curvature()

        plot_data["pos_x_setpoint"].append(pos[0])
        plot_data["pos_y_setpoint"].append(pos[1])
        plot_data["max_velocity"].append(traj_planner.MAX_VELOCITY_var)
        plot_data["pos_y_real"].append(traj_planner.pos)
        plot_data["velocity"].append(traj_planner.vel)
        plot_data["acceleration"].append(sqrt(traj_planner._calc_normal_acceleration()**2 + traj_planner._calc_tangential_acceleration()**2))
        plot_data["acceleration_tangential"].append(traj_planner._calc_tangential_acceleration())
        plot_data["acceleration_normal"].append(traj_planner._calc_normal_acceleration())
        
        plot_data["curvature"].append(curvature) 
        plot_data["time"].append(s*dt)

    plotter(plot_data)


if __name__ == "__main__":
    main(4)
