import numpy as np
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_snippet(notes, size):
    cut = len(notes)%size
    # cut off values from end of trajectory to match sizes
    if cut > 0:
        notes = notes[:-cut]
    snippets = np.split(notes, len(notes)/size)
    return snippets

def distance_loss(x, y):
    return np.linalg.norm(x-y, ord=1)


def gradient_loss(x, y):
    grad_diff = np.gradient(x) - np.gradient(y)
    return np.linalg.norm(grad_diff, ord=2)
 
def curve_loss(x, y):
    curve_diff = np.gradient(np.gradient(x)) - np.gradient(np.gradient(y))
    return np.linalg.norm(curve_diff, ord=2)


def match_motifs(motifs, snippets):
    matches = []
    for snippet in snippets:
        losses = []
        print("Snippet")
        # Calculate losses for this snippet compared to each motif
        for motif in motifs:
            # TODO: weight parameters?
            A = 1
            B = 1
            C = 1
            loss = A*distance_loss(snippet, motif) + B*gradient_loss(snippet, motif) + C*curve_loss(snippet, motif)
            losses.append(loss)
        print(losses)
        match = np.argmin(losses)
        matches.append(match)
    return matches

def plot_notes_traj(traj, motifs, matches, filename):
    num_timesteps = len(traj)
    x = np.arange(0, num_timesteps)
    x2 = np.arange(0, len(motifs[0])*len(matches))
    music = []
    for match in matches:
        music += motifs[match].tolist() 

    plt.clf()
    plt.figure(figsize=(20, 5))
    plt.plot(x, traj, '-_b', x2, music, '-_m')
    plt.legend(['original trajectory', 'generated notes'])
    plt.ylim(60, 100)
    plt.xlabel('time')
    plt.ylabel('pitch')
    plt.savefig(filename)



if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python find_motifs.py [traj filename] [music filename]')


    traj_file = sys.argv[1]
    music_file = sys.argv[2]
    traj = np.loadtxt(traj_file)
    notes = np.loadtxt(music_file)

    size_motif = 5
    motifs = extract_snippet(notes, size_motif)
    snippets = extract_snippet(traj, size_motif)

    matches = match_motifs(motifs, snippets)
    print(matches)
    print(len(matches))
 
    # Plot trajectory overlaid with selected series of motifs
    plot_notes_traj(traj, motifs, matches, 'notes-traj.pdf')

