import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sin, cos, tan, sqrt, asin, acos,pi, exp
import matplotlib.pyplot as plt


def to_rad(theta):
    return theta*np.pi/180

def task1():
    m_a = 1
    m_b = 3
    m_d = 20
    R_b = 20
    R_d = 20
    i_bx = 18
    r_b = 16
    fk = 64
    phi = 0.6
    g = 9.8

    ground_height = 10
    v_a = []
    s_all = []
    for i in range(0,ground_height):
        s = ground_height - i
        s_all.append(s)
        v_a_2 = (m_a*s + m_d*g*((fk+R_b-sqrt(fk**2 - R_b**2))))
        v_a_2 /= (0.5*m_a+0.5*m_b*i_bx/r_b**2 + 0.5*m_d*R_b**2/r_b**2 + 0.25*m_d*R_b**2/((R_d**2)*(r_b**2)))
        v_a.append(sqrt(v_a_2))
    plt.plot(s_all,v_a, label="Velocity of A")
    plt.ylabel('V_a -- (m/s)')
    plt.xlabel('S -- m')
    plt.legend()
    plt.show()

def task2():
    m = 1
    b = 0.001
    dot_eps0 = 3
    g = 9.8
    A = -0.953 * dot_eps0/b + 5.157/(b**2)
    B = 2*(b**2)*1.049*A/(m*sqrt(3))
    C = 2/sqrt(3) * 1.1-A*b*b
    D = g - 2*b*5.41/(sqrt(3) * m*b)
    E = B-C
    t = np.linspace(0,5,1000)
    epsilon = lambda t: A*(exp(-1.049*b*t) - 1) + 5.41*t/b
    x = lambda t: 0.9087 *E/b*exp(-1.049*b*t) + D*t*t/2 + 0.953*E/b*t #- 0.908*E/(b*b)
    all_epsilon = []
    all_x = []
    all_x_axis_x = []
    all_y_axis_x = []
    all_x_axis_eps = []
    all_y_axis_eps = []
    for i in t:
        all_epsilon.append(epsilon(i))
        all_x.append(x(i)/500)
        all_x_axis_eps.append(epsilon(i)*cos(to_rad(30)))
        all_y_axis_eps.append(epsilon(i)*sin(to_rad(30)))
        all_x_axis_x.append(x(i)*cos(to_rad(30)))
        all_y_axis_x.append(x(i)*sin(to_rad(30)))
    plt.plot(t, all_epsilon, label="Epsilon vs time")
    plt.plot(t, all_x, label="X vs time")
    plt.ylabel('Epsilon and X -- (m)')
    plt.xlabel('time -- s')
    plt.legend()
    plt.show()
    plt.plot(t, all_x_axis_x, label="X")
    plt.plot(t, all_y_axis_x, label="Y")
    plt.ylabel('Coordinates -- (m)')
    plt.xlabel('time -- s')
    plt.legend()
    plt.show()

    plt.plot(t, all_x_axis_eps, label="X")
    plt.plot(t, all_y_axis_eps, label="Y")
    plt.ylabel('Coordinates -- (m)')
    plt.xlabel('time -- s')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tasks = [task1, task2]

    tasks[int(sys.argv[1])-1]()