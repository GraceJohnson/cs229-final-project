from music21 import converter, instrument, note, chord, stream, pitch
import numpy as np
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_midi(filename):
    midi = converter.parse(filename)
    notes_i = []

    print("Parsing %s" % filename)

    notes_to_parse = None

    try: # file has instrument parts
        notes_to_parse = midi[0].recurse()
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            #notes_i.append(str(element.pitch))
            # Append the midi value
            notes_i.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            notes_i.append(element.pitches[-1].midi) # take the note with the highest octave? This is a modification

    return notes_i

def plot_notes_traj(traj, sample, filename):
    num_timesteps = len(traj)
    x = np.arange(0, num_timesteps)

    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(x, traj, '-_k', x, sample, '-_r')
    plt.legend(['original trajectory', 'LSTM generated notes'], loc='upper left')
    plt.ylim(50, 90)
    plt.xlabel('timesteps')
    plt.ylabel('pitch')
    plt.savefig(filename)



if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python find_motifs.py [traj filename] [midi filename] [out filename]')

    traj_file = sys.argv[1]
    sample_file = sys.argv[2]
    out_file = sys.argv[3]

    sample = read_midi(sample_file)
    traj = np.loadtxt(traj_file)
    traj = traj[:len(sample)]
    sample = sample[:len(traj)]

    plot_notes_traj(traj, sample , out_file)

