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
        ax = plt.subplot(221)
        ax.plot(time, plotters[particle]["position"])
        ax.legend(["x(t)", "y(t)"], loc='upper right')
        ax.set_title('Position')
        ax.set(xlabel='time', ylabel='Postion (x(t),y(t))')

        ax = plt.subplot(222)
        ax.plot(time, plotters[particle]["velocity"])
        ax.legend(["v(t), v_x(t)", "v_y(t)"], loc='upper right')
        ax.set_title('Velocity')
        ax.set(xlabel='time', ylabel='velocity')

        ax = plt.subplot(223)
        ax.plot(time, plotters[particle]["acceleration"]) 
        ax.plot(time, plotters[particle]["acceleration_normal"])
        ax.plot(time, plotters[particle]["acceleration_tangential"])
        ax.legend(["a(t)", "an(t)", "at(t)"], loc='upper right')
        ax.set_title('Acceleration')
        ax.set(xlabel='time', ylabel='acceleration')
        # print(len(plotters[particle]["acceleration"]), len(plotters[particle]["acceleration_normal"]), len(plotters[particle]["acceleration_tangential"]))

        if(len(plotters[particle].keys()) > 5):
            ax = plt.subplot(224)
            ax.plot(time, plotters[particle]['rho'])
            ax.legend(["rho(t)"], loc='upper right')
            ax.set_title('Curvature coefficient')
            ax.set(xlabel='time', ylabel='rho(t)')

        plt.tight_layout()
        fig.canvas.set_window_title('Point {:}'.format(particle))
        plt.show()
    


def task1():
    # For one point X  (coordinate form (x,y))
    plotters = {"P":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": [], "rho": []}}

    time = np.linspace(-5, 5, 1000)

    for t in time:
        x = 3*t
        y = 4*(t**2) + 1
        y_x = 4/9*(x**2)+1
        vx = 3
        vy = 8*t
        v = sqrt(vx**2 + vy**2)

        an = (4096*t)/(v)
        at = v**2 * ((3/128) + (41/2)*(t**2))
        a = sqrt(an**2 + at**2)

        rho = (pow((9+64*t*t),3/2))/(24)

        plotters["P"]["position"].append([x, y])
        plotters["P"]["velocity"].append([v, vx, vy])
        plotters["P"]["acceleration"].append(a)
        plotters["P"]["acceleration_tangential"].append(at)        
        plotters["P"]["acceleration_normal"].append(an)
        plotters["P"]["rho"].append(rho)
    plotter(plotters, time)

def task2_1():
    # For B and C
    plotters = {"A":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "B":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "C":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []}}
    w = 1
    alpha = 0
    AB = 80
    OA = 25
    AC = 20
    PO = 25

    slope_PB = tan(to_rad(30))
    y_intercept_PB = PO 
    time = np.linspace(0, 1, 1000)

    for t in time:
        phi = w*t
        x_a, y_a = OA*sin(phi), OA*cos(phi)
        vx_a, vy_a = OA*w*cos(phi), -OA*w*sin(phi)
        v_a = OA*w
        ax_a, ay_a = OA*alpha*cos(phi) - OA*w**2*sin(phi), -OA*alpha*sin(phi) - OA*w**2*cos(phi) 
        a_a = sqrt(ax_a**2 + ay_a**2)
        at_a = (OA**2)*alpha
        an_a = sqrt(a_a**2-at_a**2)

        plotters["A"]["position"].append([x_a, y_a])
        plotters["A"]["velocity"].append([v_a, vx_a, vy_a])
        plotters["A"]["acceleration"].append([a_a])
        plotters["A"]["acceleration_tangential"].append(at_a)        
        plotters["A"]["acceleration_normal"].append(an_a)

        eq1_cirAB = (lambda x: (x[0] - x_a)**2 + (x[1] - y_a)**2 - (AB)**2)
        eq2_linePB = (lambda x: x[1] - slope_PB*x[0] - y_intercept_PB)
        eqs = (lambda x: [eq1_cirAB(x), eq2_linePB(x)])
        x_b, y_b = fsolve(eqs, [PO/tan(to_rad(30)),0])

        bsi_OBA = asin(OA*sin(np.pi/2-phi)/AB)
        theta_OICB = phi - bsi_OBA
        AIC = 1*AB/sin(theta_OICB)
        w_ab = v_a/AIC
        
        BIC = sqrt(AIC**2 - AB**2)
        v_b = w_ab*BIC
        # Not sure
        an_b = (w_ab**2) * (BIC)
        at_b = alpha*(BIC**2)
        a_b = sqrt(an_b**2 + at_b**2)
        plotters["B"]["position"].append([x_b, y_b])
        plotters["B"]["velocity"].append([v_b])
        plotters["B"]["acceleration"].append(a_b)
        plotters["B"]["acceleration_tangential"].append(at_b)        
        plotters["B"]["acceleration_normal"].append(an_b)

        eq1_cirAC = (lambda x: (x[0] - x_a)**2 + (x[1] - y_a)**2 - (AC)**2)
        eqs = (lambda x: [eq1_cirAC(x), eq2_linePB(x)])
        x_c, y_c = fsolve(eqs, [PO/tan(to_rad(30)),0])
        v_c = v_a + w_ab*AC
        # Not sure, should a_c = sqrt(an_c**2 + at_c**2)
        an_c = (w_ab**2) * (AC)
        at_c = alpha * (AC**2)
        a_c = an_c + at_c + a_a

        plotters["C"]["position"].append([x_c, y_c])
        plotters["C"]["velocity"].append([v_c])
        plotters["C"]["acceleration"].append(a_c)
        plotters["C"]["acceleration_tangential"].append(at_c)        
        plotters["C"]["acceleration_normal"].append(an_c)

    plotter(plotters, time)


def task2_2():
    # For B and C
    plotters = {"A":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "B":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "C":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []}}
    w = 1
    alpha = 0
    AB = 70
    OA = 35
    AC = 45
    BC = AB-AC

    time = np.linspace(0.001, 1, 1000)

    for t in time:
        phi = w*t
        x_a, y_a = -OA*sin(phi), OA*cos(phi)
        v_a = OA*w
        an_a = (w**2)*OA 
        at_a = alpha*(OA**2)
        a_a = 0 + sqrt(an_a**2 + at_a**2)

        plotters["A"]["position"].append([x_a, y_a])
        plotters["A"]["velocity"].append([v_a])
        plotters["A"]["acceleration"].append([a_a])
        plotters["A"]["acceleration_tangential"].append(at_a)        
        plotters["A"]["acceleration_normal"].append(an_a)
        

        beta = asin(OA*cos(phi)/AB)
        # alpha = np.pi/2-beta+phi
        # x_b, y_b = sin(alpha)*AB/cos(phi), 0
        x_b, y_b = sqrt(AB**2 - y_a**2)-abs(x_a), 0
        # v_b = v_a*cos(alpha)/cos(beta)
        v_b = OA*sin(phi)*w
        PB = x_b/tan(phi)
        w_ab = v_b/PB
        an_b = PB*(w_ab**2)
        at_b = (PB**2)*alpha
        a_b =  sqrt(an_b**2 + at_b**2)
        plotters["B"]["position"].append([x_b, y_b])
        plotters["B"]["velocity"].append([v_b])
        plotters["B"]["acceleration"].append(a_b)
        plotters["B"]["acceleration_tangential"].append(at_b)        
        plotters["B"]["acceleration_normal"].append(an_b)

        x_c, y_c = x_b - (BC*x_b/AB), BC*y_a/AB
        vx_c, vy_c = (w_ab*BC*sin(beta) + v_b, - w_ab*BC*cos(beta))
        v_c = sqrt(vx_c**2 + vy_c**2)
        PC = v_c/w_ab
        an_c = PC*(w_ab**2)
        at_c = (PC**2) *alpha
        a_c = sqrt(an_b**2 + at_a**2)
    
        plotters["C"]["position"].append([x_c, y_c])
        plotters["C"]["velocity"].append([v_c])
        plotters["C"]["acceleration"].append(a_c)
        plotters["C"]["acceleration_tangential"].append(at_c)        
        plotters["C"]["acceleration_normal"].append(an_c)

    plotter(plotters, time)


def task3():
    # For B and C
    plotters = {"A":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "B":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "C":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []}}
    w = 1
    AB = 45
    AC = 30
    BC = AB-AC

    time = np.linspace(0, 10, 10000)

    for t in time:
        phi = w*t
        x_a, y_a = 0, 22.5+10*sin(np.pi/5*t)
        v_a = 2*np.pi*cos(np.pi/5*t)

        x_b, y_b = sqrt(AB**2 - y_a**2), 0

        PA = x_b
        w_ab = v_a/PA
        an_a =  w_ab*w_ab*PA
        at_a = -2/5*(np.pi**2)*sin(np.pi/5*t)
        a_a = sqrt(an_a**2 + at_a**2)
        alpha = sqrt(abs(at_a)/PA)
        plotters["A"]["position"].append([x_a, y_a])
        plotters["A"]["velocity"].append([v_a])
        plotters["A"]["acceleration"].append([a_a])
        plotters["A"]["acceleration_tangential"].append(at_a)        
        plotters["A"]["acceleration_normal"].append(an_a)
        

        v_b = w_ab*y_a
        PB = y_a
        an_b = w_ab*w_ab*PB
        at_b = (alpha**2)*PB
        a_b = sqrt(an_b**2 + at_b**2)
        
        plotters["B"]["position"].append([x_b, y_b])
        plotters["B"]["velocity"].append([v_b])
        plotters["B"]["acceleration"].append(a_b)
        plotters["B"]["acceleration_tangential"].append(at_b)        
        plotters["B"]["acceleration_normal"].append(an_b)

        x_c, y_c = BC*x_b/AB, BC*y_a/AB
        PC = sqrt((y_a-y_c)**2 + (x_b-x_c)**2)
        v_c = PC*w_ab
        an_c = w_ab*w_ab*PC
        at_c = (alpha**2)*PC
        a_c =  sqrt(an_c**2 + at_c**2)
        plotters["C"]["position"].append([x_c, y_c])
        plotters["C"]["velocity"].append([v_c])
        plotters["C"]["acceleration"].append(a_c)
        plotters["C"]["acceleration_tangential"].append(at_c)        
        plotters["C"]["acceleration_normal"].append(an_c)
    plotter(plotters, time)


if __name__ == "__main__":
    # if(len(sys.argv) < 2):
    #     exit("python3 task.py [1,2,3,4]")
    tasks = [task1, task2_1, task2_2, task3]



    tasks[int(sys.argv[1])-1]()