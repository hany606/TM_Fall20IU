import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sin, cos, tan, sqrt, asin, acos,pi

def to_rad(theta):
    return theta*np.pi/180

def task1():
    AA2 = 2
    A2B = 2
    AB = AA2 + A2B
    BC = 3
    CD = 2.5
    AD = AB + BC + CD
    DE = 2
    BE = BC + CD + DE
    EE2 = 2.5
    E2F = 2
    EF = EE2 + E2F
    AF = AB + BE + EF
    
    P1 = 12
    P2 = 18
    M1 = 36
    q = 1.4
    Q = q*BE

    alpha = 45
    theta = to_rad(90-alpha)
    beta = to_rad(60)

    A = np.array([
                    [-1, cos(theta), 0, 0, 0],
                    [0, sin(theta), 1, 1, 1],
                    [0, 0, AF, AB, AD],
                    # [1, -cos(theta), 0, 0, 0],    This equation makes the rank of A not a full rank matrix
                    [0, 0, 0, AB, 0],
                    [0, sin(theta), 0, 1, 0]
                 ])
    B = np.array([
                    P2*cos(beta),
                    Q + P2*sin(beta) + P2,
                    M1 + P1*(AA2) + Q*(BE/2+AB),
                    # 0,
                    q*BC*(BC/2+AB) + P1*AA2,
                    q*BC + P1
                 ])
    # print(np.linalg.matrix_rank(A))
    sol = np.linalg.solve(A,B)
    # print(sol)
    print("RA: {:}, \nRBx: {:}, \nRBy: {:}, \nRDy=RD: {:}, \nRF=RFy: {:}".format(sol[1], sol[0], sol[3], sol[4], sol[2]))
    print("Notice that the negative sign means that the force is in the opposite direction regarding the assumed one")

def task2():
    G = 18
    P = 30
    a,b,c = 4, 4.5, 3.5

    A = np.array([
                    [0,0,0,0,0,1,0,0,-1,0,0,0],
                    [0,0,1,0,0,1,0,0,0,0,0,0],
                    [0,-a,0,-b,a,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,1],
                    [0,0,0,0,b,0,-b,a,0,0,a,0],
                    # [0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,1,0,0,0,0,0,-b/a],
                    [b,-a,0,0,-a,0,0,0,0,0,b,0],
                    [-1,0,0,-1,0,0,1,0,0,1,0,0],
                    [0,1,0,0,-1,0,0,-1,0,0,1,0],
                    # [0,0,1,0,0,1,0,0,1,0,0,1],
                    # [0,0,1,0,0,0,0,0,0,0,0,1],
                    # [0,0,0,0,0,0,0,0,1,0,0,1],
                    [b,0,0,0,0,0,0,a,0,b,-a,0],
                    # [0,-c,0,0,c,-b,0,c,-b,0,-c,0]
                    [1,0,a/c,-1,0,a/c,1,0,0,1,0,0],
                 ])
    B = np.array([
                    G/2-P*c/b,
                    G/2,
                    0,
                    P*c/b+G/2,
                    G/2,
                    P*a,
                    # P*c/b+G/2,
                    G/2,
                    0,
                    0,
                    -P,
                    # G
                    # G/2-P*c/b,
                    # G*a/2
                    P*a,
                    G*a/(c*2)
                 ])
    # print(np.shape(A))
    # print(np.linalg.matrix_rank(A))
    sol = np.linalg.solve(A,B)
    # print(sol)
    names = ["R_alpha_x", "R_alpha_y", "R_alpha_z",
         "R_beta_x", "R_beta_y", "R_beta_z",
         "R_gamma_x", "R_gamma_y", "R_gamma_z",
         "R_ksi_x", "R_ksi_y", "R_ksi_z"]
    for i,s in enumerate(sol):
        print("{:}: {:},".format(names[i],s))

    print("Notice that the negative sign means that the force is in the opposite direction regarding the assumed one")



if __name__ == "__main__":
    tasks = [task1, task2]

    tasks[int(sys.argv[1])-1]()