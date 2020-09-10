import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve
from math import sin, cos, tan, sqrt, asin, acos


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
                "B":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "C":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "D":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "E":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "F":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "AB":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "AO1":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "BO2":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "AC":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "CB":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "CD":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "EF":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
                "FO3":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []}}
    
    time = np.linspace(0, 5, 1000)

    w_O1A = 2
    phi0 = to_rad(60)
    a, b, c, d, e = 56, 10, 26, 16, 25
    O1A, O2B, O3F, AB, BC, CD, CE, EF = 21, 25, 20, 54, 52, 69, 35, 32

    AC = 1/3*CD


    x_o1, y_o1 = 0,0 
    x_o2, y_o2 = a, -c
    x_o3, y_o3 = a+b, d+e

    for t in time:
        phi = w_O1A*t + phi0
        x_a, y_a = O1A*cos(phi), O1A*sin(phi)

        # find intersections of the geometric objects to find the intersection points
        eq1_cirBO2 = (lambda x: (x[0] - x_o2)**2 + (x[1] - y_o2)**2 - (O2B)**2)
        eq2_cirAB  = (lambda x: (x[0] - x_a)**2 + (x[1] - y_a)**2 - (AB)**2)
        eqs = (lambda x: [eq1_cirBO2(x), eq2_cirAB(x)])
        x_b, y_b = fsolve(eqs, [0,0])

        eq1_cirAC = (lambda x: (x[0] - x_a)**2 + (x[1] - y_a)**2 - (AC)**2)
        eq2_cirBC  = (lambda x: (x[0] - x_b)**2 + (x[1] - y_b)**2 - (BC)**2)
        eqs = (lambda x: [eq1_cirAC(x), eq2_cirBC(x)])
        x_c, y_c = fsolve(eqs, [0,0])
        
        eq1_cirDC = (lambda x: (x[0] - x_c)**2 + (x[1] - y_c)**2 - (CD)**2)
        eq2_lineD  = (lambda x: x[1]-d)
        eqs = (lambda x: [eq1_cirDC(x), eq2_lineD(x)])
        x_d, y_d = fsolve(eqs, [0,0])
        
        # E
        CD_vec = [x_d-x_c, y_d-y_c]
        CD_vec_mag = sqrt(CD_vec[0]**2 + CD_vec[1]**2)

        x_e, y_e = x_c + CE*(CD_vec[0]/CD_vec_mag), y_c + CE*(CD_vec[1]/CD_vec_mag)

        eq1_cirO3F = (lambda x: (x[0] - x_o3)**2 + (x[1] - y_o3)**2 - (O3F)**2)
        eq2_cirEF  = (lambda x: (x[0] - x_e)**2 + (x[1] - y_e)**2 - (EF)**2)
        eqs = (lambda x: [eq1_cirO3F(x), eq2_cirEF(x)])
        x_f, y_f = fsolve(eqs, [0,0])

        plotters["A"]["position"].append([x_a, y_a])
        plotters["B"]["position"].append([x_b, y_b])
        plotters["C"]["position"].append([x_c, y_c])
        plotters["D"]["position"].append([x_d, y_d])
        plotters["E"]["position"].append([x_e, y_e])
        plotters["F"]["position"].append([x_f, y_f])

        vx_a, vy_a = -w_O1A*y_a, w_O1A*x_a

        r_BA = [x_b-x_a, y_b-y_a]
        r_BO2 = [x_b-x_o2, y_b-y_o2]
        r_CA = [x_c-x_a, y_c-y_a]
        r_DC = [x_d-x_c, y_d-y_c]
        r_EC = [x_e-x_c, y_e-y_c]
        r_FE = [x_f-x_e, y_f-y_e]
        r_FO3 = [x_f-x_o3, y_f-y_o3]

        w_ABC = -vy_a*r_BO2[1]/(r_BO2[0]*(-r_BA[1]+(r_BA[0]*r_BO2[1]/r_BO2[0])))
        w_BO2 = vy_a/r_BO2[0] + w_ABC*r_BA[0]/r_BO2[0]

        vx_b, vy_b = vx_a - w_ABC*r_BA[1], w_BO2*r_BO2[0]

        vx_c, vy_c = vx_a - w_ABC*r_CA[1], vx_a + w_ABC*r_CA[0] 
        
        w_CD = - vy_c/r_DC[0]
        vx_d, vy_d = vx_c - w_CD*r_DC[1], 0

        vx_e, vy_e = vx_c - w_CD*r_EC[1], vy_c + w_CD*r_EC[0]

        w_EF = -vy_e*r_FO3[1]/(r_FO3[0]*(-r_FE[1]+(r_FE[0]*r_FO3[1]/r_FO3[0])))
        w_FO3 = vy_e/r_FO3[0] + w_EF*r_FE[0]/r_FO3[0]

        vx_f, vy_f = vx_e - w_EF*r_FE[1], w_FO3*r_FO3[0]


        plotters["A"]["velocity"].append([vx_a, vy_a])
        plotters["B"]["velocity"].append([vx_b, vy_b])
        plotters["C"]["velocity"].append([vx_c, vy_c])
        plotters["D"]["velocity"].append([vx_d, vy_d])
        plotters["E"]["velocity"].append([vx_e, vy_e])
        plotters["F"]["velocity"].append([vx_f, vy_f])

        plotters["AB"]["angular_velocity"].append([w_ABC])
        plotters["AC"]["angular_velocity"].append([w_ABC])
        plotters["CB"]["angular_velocity"].append([w_ABC])
        plotters["AO1"]["angular_velocity"].append([w_O1A])
        plotters["BO2"]["angular_velocity"].append([w_BO2])
        plotters["CD"]["angular_velocity"].append([w_CD])
        plotters["EF"]["angular_velocity"].append([w_EF])
        plotters["FO3"]["angular_velocity"].append([w_FO3])

    plotter(plotters, time)


def task2():
    # For B and C
    plotters = {"A":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
               "C":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []},
               "M":{"position": [], "velocity": [], "angular_velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "angular_acceleration": []}}
    w_B = 2
    e_B = 3.7
    OM0 = 40
    MM0 = 5
    CM0 = OM0*cos(to_rad(15))
    OC = sqrt(OM0**2 - MM0**2)

    time = np.linspace(0, 1, 1000)

    for t in time:
        w_A = w_B*sin(to_rad(45))/sin(to_rad(15))
        v_c = w_B*OC*sin(to_rad(45))
        e_a_n = w_A*w_B*sin(to_rad(60))
        e_a_t = e_B*sin(to_rad(45))/sin(to_rad(15))
        e_a = sqrt(e_a_n**2 + e_a_t**2)

        plotters["A"]["angular_velocity"].append([w_A])
        plotters["A"]["angular_acceleration"].append([e_a])
        

        v_m = w_A*MM0*sin(to_rad(30))

        OM = sqrt(MM0**2 + OM0**2 - 2* MM0*OM0*cos(to_rad(75)))

        theta = asin(sin(to_rad(75))*MM0/OM)

        a_mc_t = e_a_t*OM*sin(theta)
        a_mc_n = e_a_n*OM
        a_c = w_A*v_m

        a_m = sqrt(a_c**2 + a_mc_n**2 - 2*a_c*a_mc_n*cos(theta) + a_mc_t**2)

        plotters["C"]["velocity"].append([v_c])
        plotters["C"]["acceleration"].append([a_c])

        plotters["M"]["velocity"].append([v_m])
        plotters["M"]["acceleration"].append([a_m])
        
        

    plotter(plotters, time)

if __name__ == "__main__":
    tasks = [task1, task2]

    tasks[int(sys.argv[1])-1]()