# %matplotlib notebook
from music21 import converter, instrument, note, stream, chord
# import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from IPython.display import clear_output
import mido
import glob, pickle
import sys
import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        notes_i = []

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes_i.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes_i.append(str(element.pitches[-1])) # take the note with the highest octave? This is a modification
        
        # trim out the excess to standardize the length of the musical piece
        # desired_length = len(notes_i) - (len(notes_i) % 50) # 50 will be our input length
        notes.append(notes_i)

    assert len(notes) == len(glob.glob("midi_songs/*.mid"))
    
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

if __name__ == '__main__':
	get_notes()
