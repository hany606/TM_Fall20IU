import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi

# f(x) = asin(bx + c) + d, where a is the amplitude, b is the period, c is phase shift(x-axis, previous x), d is vertical shift(y-axis, previous y)
# def func_sin(t):
#     if()

def test():
    # amplitude
    string_length = 42#69
    t = np.linspace(0,15, 1000)
    y_all = []
    for i in t:
        a = 5
        shift = pi + 1.5
        squeez_y = 0.7*i
        y = string_length - squeez_y*abs(string_length/(i*a-shift)*sin(i*a-shift))#abs(string_length*sin(5*i)) #func_sin(t)
        y_all.append(y)
    experiment = 3
    yoyo_y = np.load(f"Exp{experiment}{1}_cm.npy")
    print(len(yoyo_y))
    length = len(yoyo_y)-100
    plt.plot(t[:length], y_all[:length], label="Vertical position -- simulation")
    plt.plot(t[:length], yoyo_y[:length], label="Vertical position -- real")
    plt.ylabel("Position -- cm")
    plt.xlabel("time -- sec")
    plt.legend()
    plt.title("Yo-yo simulation")
    plt.savefig("Yoyo_sim_exp31.png")
    plt.show()

if __name__ == "__main__":
    test()