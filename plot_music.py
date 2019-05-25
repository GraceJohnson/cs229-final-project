import numpy as np
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_notes(traj, filename):
    num_timesteps = len(traj)
    x = np.arange(0, num_timesteps)
    plt.clf()
    plt.figure(figsize=(20, 5))
    plt.plot(x, traj, '-_b')
    plt.ylim(60, 100)
    plt.xlabel('time')
    plt.ylabel('pitch')
    plt.savefig(filename)

if __name__ == '__main__':

    music_file = sys.argv[1]
    notes = np.loadtxt(music_file)
    plot_notes(notes, 'notes.pdf')



