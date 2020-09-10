import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve
from math import sin, cos, tan, sqrt, asin, acos,pi


def to_rad(theta):
    return theta*np.pi/180

def plotter(plotters, time):
    # plot position
    # plot velocity
    # plot acceleration with normal and tangential
    for particle in plotters.keys():
        print(particle)
        fig = plt.figure()
        if(len(plotters[particle]["position"]) > 0):
            ax = plt.subplot(221)
            ax.plot(time, plotters[particle]["position"])
            ax.legend(["x(t)", "y(t)"], loc='upper right')
            ax.set_title('Position')
            ax.set(xlabel='time', ylabel='Postion (x(t),y(t))')
        
        if(len(plotters[particle]["velocity"])  > 0):
            ax = plt.subplot(222)
            ax.plot(time, plotters[particle]["velocity"])
            if(len(plotters[particle]["angular_velocity"])  > 0):
                ax.plot(time, plotters[particle]["angular_velocity"])
            ax.legend(["v_x(t)", "v_y(t), w(t)"], loc='upper right')
            ax.set_title('Velocity')
            ax.set(xlabel='time', ylabel='velocity')
    
        if(len(plotters[particle]["angular_velocity"])  > 0 and len(plotters[particle]["velocity"])  == 0):
            ax = plt.subplot(221)
            ax.plot(time, plotters[particle]["angular_velocity"])
            ax.legend(["w(t)"], loc='upper right')
            ax.set_title('Angular_Velocity')
            ax.set(xlabel='time', ylabel='velocity')
            
        if(len(plotters[particle]["acceleration"])  > 0):
            ax = plt.subplot(223)
            ax.plot(time, plotters[particle]["acceleration"])
            if(len(plotters[particle]["acceleration_normal"])  > 0):
                ax.plot(time, plotters[particle]["acceleration_normal"])
            if(len(plotters[particle]["acceleration_tangential"])  > 0):
                ax.plot(time, plotters[particle]["acceleration_tangential"])
            ax.legend(["a(t)", "an(t)", "at(t)"], loc='upper right')
            ax.set_title('Acceleration')
            ax.set(xlabel='time', ylabel='acceleration')

        if(len(plotters[particle]["angular_acceleration"])  > 0):
            ax = plt.subplot(224)
            ax.plot(time, plotters[particle]["angular_acceleration"])
            ax.legend(["alpha(t)"], loc='upper right')
            ax.set_title('Angular Acceleration')
            ax.set(xlabel='time', ylabel='angular_acceleration')

        plt.tight_layout()
        fig.canvas.set_window_title('Point {:}'.format(particle))
        plt.show()
    


def task1():
    plotters = {"A":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "O":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "M":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []}}
    time = np.linspace(0, 1, 1000)


    O1O = 20
    O2A = 20
    R = 18
    O1O2 = 2*R


    for t in time:
        phi = 2*t - 0.3*(t**2)
        omega = 2 - 0.6*t
        alpha = 0.6
        OM = 6*pi*(t**2)
        dot_OM = 12*pi*t

        
        x_o = O1O*cos(phi)
        y_o = O1O*sin(phi)

        x_a = O2A*cos(phi) + O1O2
        y_a = O2A*sin(phi) + O1O2

        theta = acos(OM/(2*R))
        dot_theta = dot_OM/(-2*R*sin(theta))

        x_M = x_o + cos(theta)*OM
        y_M = y_o + sin(theta)*OM
        v_Mo_x_T = -sin(theta)*dot_theta*OM
        v_Mo_y_T = cos(theta)*dot_theta*OM
        
        v_Mo_T = sqrt(v_Mo_x_T**2 + v_Mo_y_T**2)

        v_M_T = v_Mo_T + omega*OM

        v_o_G = 0 + omega*O1O

        v_M = v_o_G + v_M_T

        a_M = 0 + alpha*O1O+2*omega*v_M_T + (omega**2)*O1O

        plotters["A"]["position"].append([x_a, y_a])
        plotters["O"]["position"].append([x_o, y_o])
        plotters["M"]["position"].append([x_M, y_M])

        plotters["M"]["velocity"].append([v_M])
        plotters["M"]["acceleration"].append([a_M])

    plotter(plotters, time)


def task2():
    # For B and C
    plotters = {"O":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
               "M":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []}}

    R = 30
    time = np.linspace(0, 1, 1000)

    for t in time:
        phi = 2*t - 0.3*(t**2)
        omega = 2 - 0.6*t
        alpha = 0.6
        OM = 75*pi*(0.1*t+0.3*(t**2))
        dot_OM = 12*pi*t

        
        x_o = R*cos(phi)
        y_o = R*sin(phi)

        theta = acos(min(0.999,OM/(2*R)))
        dot_theta = dot_OM/(-2*R*sin(theta))

        x_M = x_o + cos(theta)*OM
        y_M = y_o + sin(theta)*OM
        
        v_Mo_x_T = -sin(theta)*dot_theta*OM
        v_Mo_y_T = cos(theta)*dot_theta*OM
        
        v_Mo_T = sqrt(v_Mo_x_T**2 + v_Mo_y_T**2)

        v_M_T = v_Mo_T + omega*OM

        v_o_G = 0 + omega*R

        v_M = v_o_G + v_M_T

        a_M = 0 + alpha*R+2*omega*v_M_T + (omega**2)*R

        plotters["O"]["position"].append([x_o, y_o])
        plotters["M"]["position"].append([x_M, y_M])

        plotters["M"]["velocity"].append([v_M])
        plotters["M"]["acceleration"].append([a_M])

        

    plotter(plotters, time)

if __name__ == "__main__":
    tasks = [task1, task2]

    tasks[int(sys.argv[1])-1]()