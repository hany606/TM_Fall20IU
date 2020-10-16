import numpy as np
import matplotlib.pyplot as plt

def original_experiments():
  for exp_i in range(5):
    for exp_j in range(3):
      yoyo_y = np.load(f"Exp{exp_i+1}{exp_j+1}_cm.npy")
      t_steps = [i for i in range(len(yoyo_y))]
      plt.plot(t_steps, yoyo_y, label="Y position")
      plt.title(f"Exp{exp_i+1}{exp_j+1} Y position")
      plt.xlabel("timesteps -- steps")
      plt.ylabel("Y pos -- ~cm")
      plt.legend()
      plt.savefig(f"Exp{exp_i+1}{exp_j+1}_fig.png")
      # plt.show()
      plt.clf()

def original_experiments_compound():
  for exp_i in range(5):
    for exp_j in range(3):
      yoyo_y = np.load(f"Exp{exp_i+1}{exp_j+1}_cm.npy")
      t_steps = [i for i in range(len(yoyo_y))]
      plt.plot(t_steps, yoyo_y, label=f"Y position - Repeat:{exp_j+1}")
      plt.title(f"Exp{exp_i+1} Y position")
      plt.xlabel("timesteps -- steps")
      plt.ylabel("Y pos -- ~cm")
      plt.legend()
    plt.savefig(f"Exp{exp_i+1}_fig.png")
    # plt.show()
    plt.clf()

def calculate_oscilations():
  for exp_i in range(5):
    for exp_j in range(3):
      yoyo_y = np.load(f"Exp{exp_i+1}{exp_j+1}_cm.npy")
      maxima = []
      minima = []
      thresh = 1
      for i in range(thresh,len(yoyo_y)-thresh):
        prev = yoyo_y[i-thresh]
        current = yoyo_y[i]
        next = yoyo_y[i+thresh]
        if(current > prev and current > next):
          maxima.append(current)
        if(current < prev and current < next):
          minima.append(current)
      print(f"Exp{exp_i+1}{exp_j+1}: Num. oscilations: {min(len(maxima), len(minima))}")
if __name__ == "__main__":
    original_experiments_compound()
    # calculate_oscilations()