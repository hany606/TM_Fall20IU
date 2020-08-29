import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve
from math import sin, cos, tan, sqrt, asin, acos


def to_rad(theta):
    return theta*np.pi/180

def plotter(data):
    # plot position
    # plot velocity
    # plot acceleration with normal and tangential
    pass

def task1(t):
    # For one point X  (coordinate form (x,y))
    pass

def task2_1(t):
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

    slope_PB = -1/tan(to_rad(30))
    y_intercept_PB = PO 
    time = np.linspace(0, 3, 500)

    for t in time:
        phi = w*t
        x_a, y_a = OA*sin(phi), OA*cos(phi)
        vx_a, vy_a = OA*w*cos(phi), -OA*w*sin(phi)
        v_a = OA*w
        ax_a, ay_a = OA*alpha*cos(phi) - OA*w**2*sin(phi), -OA*alpha*sin(phi) - OA*w**2*cos(phi) 
        a_a = sqrt(ax_a**2 + ay_a**2)

        plotters["A"]["position"].append([x_a, y_a])
        plotters["A"]["velocity"].append([vx_a, vy_a, v_a])
        plotters["A"]["acceleration"].append([ax_a, ay_a, a_a])
        plotters["A"]["acceleration_tangential"].append((OA**2)*alpha)        
        plotters["A"]["acceleration_normal"].append(sqrt(a_a**2+(OA*alpha)**2))

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



def task2_2(t):
    # For B and C
    plotters = {"A":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "B":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "C":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []}}
    w = 1
    alpha = 0
    AB = 70
    OA = 35
    AC = 45

    time = np.linspace(0, 3, 500)

    for t in time:
        phi = w*t
        x_a, y_a = OA*sin(phi), OA*cos(phi)
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
        alpha = np.pi/2-beta+phi
        # x_b, y_b = sin(alpha)*AB/cos(phi), 0
        x_b, y_b = sqrt(OA**2+AB**2)-OA*cos(phi), 0
        # v_b = v_a*cos(alpha)/cos(beta)
        v_b = OA*sin(phi)*w
        PB = cos(phi)*x_b/sin()
        # w_ab = 
        an_b = OA*(w**2)*cos(phi)
        at_b = OA*alpha*sin(phi)
        a_b =  sqrt(an_b**2 + at_a**2)
        plotters["B"]["position"].append([x_b, y_b])
        plotters["B"]["velocity"].append([v_b])
        plotters["B"]["acceleration"].append(a_b)
        plotters["B"]["acceleration_tangential"].append(at_b)        
        plotters["B"]["acceleration_normal"].append(an_b)

        x_c, y_c = ?, ?
        v_c = ?
        an_c = ?
        at_c = ?
        a_c =  ?
        plotters["C"]["position"].append([x_c, y_c])
        plotters["C"]["velocity"].append([v_c])
        plotters["C"]["acceleration"].append(a_c)
        plotters["C"]["acceleration_tangential"].append(at_c)        
        plotters["C"]["acceleration_normal"].append(an_c)



def task3(t):
    # For B and C
    plotters = {"A":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "B":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []},
                "C":{"position": [], "velocity": [], "acceleration": [], "acceleration_normal": [], "acceleration_tangential": []}}
    w = 1
    alpha = 0
    AB = 45
    AC = 30
    BC = AB-AC

    time = np.linspace(0, 10, 500)

    for t in time:
        phi = w*t
        x_a, y_a = 0, 22.5+10*sin(np.pi/5*t)
        v_a = 2*np.pi*cos(np.pi/5*t)
        an_a = ? 
        at_a = ?
        a_a = -1*2/5*(np.pi**2)*sin(np.pi/5*t)

        plotters["A"]["position"].append([x_a, y_a])
        plotters["A"]["velocity"].append([v_a])
        plotters["A"]["acceleration"].append([a_a])
        plotters["A"]["acceleration_tangential"].append(at_a)        
        plotters["A"]["acceleration_normal"].append(an_a)
        
        x_b, y_b = sqrt(AB**2 - y_a**2), 0

        w_ab = v_a/x_b

        v_b = w_ab*y_a
        an_b = ?
        at_b = ?
        a_b = ?
        plotters["B"]["position"].append([x_b, y_b])
        plotters["B"]["velocity"].append([v_b])
        plotters["B"]["acceleration"].append(a_b)
        plotters["B"]["acceleration_tangential"].append(at_b)        
        plotters["B"]["acceleration_normal"].append(an_b)

        x_c, y_c = BC*x_b/AB, BC*y_a/AB
        v_c = sqrt(y_b**2 - BC**2)*w_ab
        an_c = ?
        at_c = ?
        a_c =  ?
        plotters["C"]["position"].append([x_c, y_c])
        plotters["C"]["velocity"].append([v_c])
        plotters["C"]["acceleration"].append(a_c)
        plotters["C"]["acceleration_tangential"].append(at_c)        
        plotters["C"]["acceleration_normal"].append(an_c)


if __name__ == "__main__":
    # if(len(sys.argv) < 2):
    #     exit("python3 task.py [1,2,3,4]")
    tasks = [task1, task2_1, task2_2, task3]



    tasks[int(sys.argv[1])-1]