# Inspired and modified from https://github.com/EmanuelFeru/MotionGenerator

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin

class TrapezoidMotion():
    def __init__(self, max_velocity, max_acceleration):
        self.MAX_ACCELERATION = max_acceleration
        self.MAX_VELOCITY = max_velocity
        
        self.pos = 0
        self.vel = 0
        self.acc = 0
        self.oldPos = 0
        self.oldPosRef = 0
        self.oldVel = 0
        

        self.dBrk = 0
        self.dAcc = 0
        self.dVel = 0
        self.dDec = 0
        self.dTot = 0
        

        self.tBrk = 0
        self.tAcc = 0
        self.tVel = 0
        self.tDec = 0

        self.velSt = 0

        self.oldTime = 0
        self.lastTime = 0
        self.deltaTime = 0


        self.signM = 1  # 1 = positive change, -1 = negative change
        self.shape = 1  # 1 = trapezoidal, 0 = triangular

    def update(self,posRef, time):      
        if (self.oldPosRef != posRef):  # reference changed
            # Shift state variables
            self.oldPosRef = posRef
            self.oldPos = self.pos
            self.oldVel = self.vel
            self.oldTime = self.lastTime
            
            # Calculate braking time and distance (in case is neeeded)
            self.tBrk = abs(self.oldVel) / self.MAX_ACCELERATION # same as for shape = 0
            self.dBrk = self.tBrk * abs(self.oldVel) / 2
            
            # Caculate Sign of motion
            self.signM = np.sign(posRef - (self.oldPos + np.sign(self.oldVel)*self.dBrk))
            
            if (self.signM != np.sign(self.oldVel)):  # means brake is needed
                self.tAcc = (self.MAX_VELOCITY / self.MAX_ACCELERATION)
                self.dAcc = self.tAcc * (self.MAX_VELOCITY / 2)
            else:
                self.tBrk = 0
                self.dBrk = 0
                self.tAcc = (self.MAX_VELOCITY - abs(self.oldVel)) / self.MAX_ACCELERATION
                self.dAcc = self.tAcc * (self.MAX_VELOCITY + abs(self.oldVel)) / 2
            
            # Calculate total distance to go after braking
            self.dTot = abs(posRef - self.oldPos + self.signM*self.dBrk)
            
            self.tDec = self.MAX_VELOCITY / self.MAX_ACCELERATION
            self.dDec = self.tDec * (self.MAX_VELOCITY) / 2
            self.dVel = self.dTot - (self.dAcc + self.dDec)
            self.tVel = self.dVel / self.MAX_VELOCITY
            
            if (self.tVel > 0):    # trapezoidal shape
                self.shape = 1
            else:                # triangular shape
                self.shape = 0
                # Recalculate distances and periods
                if (self.signM != np.sign(self.oldVel)):  # means brake is needed
                    self.velSt = sqrt(self.MAX_ACCELERATION*(self.dTot))
                    self.tAcc = (self.velSt / self.MAX_ACCELERATION)
                    self.dAcc = self.tAcc * (self.velSt / 2)
                else:
                    self.tBrk = 0
                    self.dBrk = 0
                    self.dTot = abs(posRef - self.oldPos)      # recalculate total distance
                    self.velSt = sqrt(0.5*self.oldVel**2 + self.MAX_ACCELERATION*self.dTot)
                    self.tAcc = (self.velSt - abs(self.oldVel)) / self.MAX_ACCELERATION
                    self.dAcc = self.tAcc * (self.velSt + abs(self.oldVel)) / 2

                self.tDec = self.velSt / self.MAX_ACCELERATION
                self.dDec = self.tDec * (self.velSt) / 2
        
        # Calculate time since last set-point change
        self.deltaTime = (time - self.oldTime)
        # Calculate new setpoint
        self.calculateTrapezoidalProfile(posRef)
        # Update last time
        self.lastTime = time

    

    def calculateTrapezoidalProfile(self, posRef):
        
        t = self.deltaTime
        
        if (self.shape):   # trapezoidal shape
            if (t <= (self.tBrk+self.tAcc)):
                self.pos = self.oldPos + self.oldVel*t + self.signM * 0.5*self.MAX_ACCELERATION*t**2
                self.vel = self.oldVel + self.signM * self.MAX_ACCELERATION*t
                self.acc = self.signM * self.MAX_ACCELERATION
            elif (t > (self.tBrk+self.tAcc) and t < (self.tBrk+self.tAcc+self.tVel)):
                self.pos = self.oldPos + self.signM * (-self.dBrk + self.dAcc + self.MAX_VELOCITY*(t-self.tBrk-self.tAcc))
                self.vel = self.signM * self.MAX_VELOCITY
                self.acc = 0
            elif (t >= (self.tBrk+self.tAcc+self.tVel) and t < (self.tBrk+self.tAcc+self.tVel+self.tDec)):
                self.pos = self.oldPos + self.signM * (-self.dBrk + self.dAcc + self.dVel + self.MAX_VELOCITY*(t-self.tBrk-self.tAcc-self.tVel) - 0.5*self.MAX_ACCELERATION*(t-self.tBrk-self.tAcc-self.tVel)**2)
                self.vel = self.signM * (self.MAX_VELOCITY - self.MAX_ACCELERATION*(t-self.tBrk-self.tAcc-self.tVel))
                self.acc = - self.signM * self.MAX_ACCELERATION
            else:
                self.pos = posRef
                self.vel = 0
                self.acc = 0
            
        else:            # triangular shape
            if (t <= (self.tBrk+self.tAcc)):
                self.pos = self.oldPos + self.oldVel*t + self.signM * 0.5*self.MAX_ACCELERATION*t**2
                self.vel = self.oldVel + self.signM * self.MAX_ACCELERATION*t
                self.acc = self.signM * self.MAX_ACCELERATION
            elif (t > (self.tBrk+self.tAcc) and t < (self.tBrk+self.tAcc+self.tDec)):
                self.pos = self.oldPos + self.signM * (-self.dBrk + self.dAcc + self.velSt*(t-self.tBrk-self.tAcc) - 0.5*self.MAX_ACCELERATION*(t-self.tBrk-self.tAcc)**2)
                self.vel = self.signM * (self.velSt - self.MAX_ACCELERATION*(t-self.tBrk-self.tAcc))
                self.acc = - self.signM * self.MAX_ACCELERATION
            else:
                self.pos = posRef
                self.vel = 0
                self.acc = 0
                
                
            
def motor_setpoint_func(t):
    # return 3
    return 1*sin(3*t+0.2)
        

if __name__ == "__main__":
    dt = 0.004
    MAX_VELOCITY = 50
    MAX_ACCELERATION = 10
    motor = TrapezoidMotion(MAX_VELOCITY,MAX_ACCELERATION)
    RUNNING_STEPS = int(4/dt)

    setpoints = [0]
    positions = [0]
    velocities = [0]
    accelerations = [0]
    time = [0]
    for t in range(1,RUNNING_STEPS):
        setpoint = motor_setpoint_func(t*dt)
        motor.update(setpoint, t*dt)
        setpoints.append(setpoint)
        time.append(t*dt)
        positions.append(motor.pos)
        velocities.append(motor.vel)
        accelerations.append(motor.acc)

    ax = plt.subplot(311)
    ax.plot(time, positions)
    ax.plot(time, setpoints)
    ax.legend(["Position","Setpoint"], loc='upper left')
    ax.set_title("Position")
    ax.set(xlabel='time', ylabel='x(t)')

    ax = plt.subplot(312)
    ax.plot(time, velocities)
    ax.set_title("Velocity")
    ax.set(xlabel='time', ylabel='v(t)')

    ax = plt.subplot(313)
    ax.plot(time, accelerations)
    ax.set_title("Acceleration")
    ax.set(xlabel='time', ylabel='a(t)')

    plt.show()

        
    
