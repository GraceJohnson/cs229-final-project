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

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    # n_vocab = len(set(np.ndarray.flatten(np.array(notes))))
    unpacked_notes = []
    
    for item in notes:
        unpacked_notes.extend(item)
    
    n_vocab = len(set(unpacked_notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)
    
    np.save("a", network_input)
    np.save("b", network_output)

    return train(model, network_input, network_output)

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
        desired_length = len(notes_i) - (len(notes_i) % 50) # 50 will be our input length
        notes.append(notes_i[0:desired_length])

    assert len(notes) == len(glob.glob("midi_songs/*.mid"))
    
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 50
    
    unpacked_notes = []
    
    for item in notes:
        unpacked_notes.extend(item)

    # get all pitch names
    pitchnames = sorted(set(item for item in unpacked_notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for song in notes:
        for i in range(0, len(song) - sequence_length, 1):
            sequence_in = song[i:i + sequence_length]
            sequence_out = song[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
    
    assert len(network_input) == len(network_output), len(network_input)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    #model.add(Dropout(0.3))
    model.add(LSTM(n_vocab))
    #model.add(Dropout(0.3))
    #model.add(LSTM(512))
    #model.add(Dense(256))
    #model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    rmsprop = optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    
    plot = live_plot()
    
    callbacks_list = [checkpoint, plot]

    history_object = model.fit(network_input, network_output, 
                               epochs=200, batch_size=64,
                               validation_split=0.2,
                               callbacks=callbacks_list)
    return history_object

class live_plot(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.index = 0
        self.epochs = []
        
        self.losses = []
        self.val_losses = []
        
        self.acc = []
        self.val_acc = []
        
        self.figure = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.epochs.append(self.index)
        
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.index += 1
        
        clear_output(wait=True)
        plt.plot(self.epochs, self.losses, label="loss")
        plt.plot(self.epochs, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig("loss.png");
        
        plt.plot(self.epochs, self.acc, label="acc")
        plt.plot(self.epochs, self.val_acc, label="val_acc")
        plt.legend()
        plt.savefig("accuracy.png");

if __name__ == '__main__':
	train_network()
