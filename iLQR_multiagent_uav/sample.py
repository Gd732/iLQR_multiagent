from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

agent1_sigmax = np.load("./solutions/agent1_sigmax.npy")
agent1_mux = np.load("./solutions/agent1_mux.npy")

sample_trajs = []
for i in range(50):
    samples = []
    for t in range(40):
        samples.append(np.random.multivariate_normal(agent1_mux[t], agent1_sigmax[t]))
    sample_traj = np.array(samples)[:,:3]
    sample_trajs.append(sample_traj)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for traj in sample_trajs:
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle='dotted', alpha=0.5)


ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.show()

