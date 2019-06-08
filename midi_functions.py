#!/usr/bin/env python3
#from mido import MidiFile
import mido
from os.path import basename
import numpy as np
import glob

def get_time_info(path='/Users/sorenh/documents/MIDI/HarryPotter.mid',v=True):
    mid = mido.MidiFile(path)
    tempo = 500000
    
    for i, track in enumerate(mid.tracks):
        for msg in track:
            msg_dict = msg.dict()
            if msg_dict['type'] == 'set_tempo':
                tempo = msg_dict['tempo']
                
            if msg_dict['type'] == 'time_signature':
                pass
            
    bpm = mido.tempo2bpm(tempo)
    ticks_per_beat = mid.ticks_per_beat
    if (v):
        print('{}: {}, secods, beats per minute = {}, ticks per beat = {}'.format(basename(path), mid.length, bpm, ticks_per_beat))
    
    return(mid.length, bpm, ticks_per_beat)
    
#doesnt work. No ones in the final matrix
def create_time_note_matrix(path='/Users/sorenh/documents/MIDI/HarryPotter.mid', ticks_per_row=10, v=True, save_file=True):
    mid = mido.MidiFile(path)
    
    matrix = []
    row = np.zeros(128)
    
    for i, track in enumerate(mid.tracks):
        for msg in track:
            m = msg.dict()
            
            if m['type'] == 'note_on':
                time_to_wait = m['time']
                if (time_to_wait > 0):
                    tick_change = 0
                    while(tick_change < time_to_wait):
                        matrix.append(row)
                        tick_change += ticks_per_row
                row[m['note']] = 1
                
            if m['type'] == 'note_off':
                time_to_wait = m['time']
                if (time_to_wait > 0):
                    tick_change = 0
                    while(tick_change < time_to_wait):
                        matrix.append(row)
                        tick_change += ticks_per_row
                row[m['note']] = 0
            
    length, bpm, ticks_per_beat = get_time_info(path, v=False)
    matrix = np.array(matrix)
    rows,d = matrix.shape
    m_ticks = rows * ticks_per_row
    minutes = length / 60
    f_ticks = minutes * bpm * ticks_per_beat
    if (v):
        print('Matrix: {} rows * {} ticks/row = {} ticks, {}: {} minutes * {} BPM * {} ticks/beat = {} ticks'.format(rows, ticks_per_row, m_ticks, basename(path), minutes, bpm, ticks_per_beat, f_ticks))
    
    if(save_file):
        filename = basename(path).split('.')[0] + '_matrix.txt'
        np.savetxt(filename,matrix,fmt='%i')
    
    return(matrix)


def create_time_note_array(path='/Users/sorenh/documents/MIDI/HarryPotter.mid', ticks_per_row=10, v=True, save_file=True):
    matrix = create_time_note_matrix(path,ticks_per_row=ticks_per_row,v=False)
    rows, d = matrix.shape
    array = np.zeros(rows)
    for i in range(0,rows):
#        print(np.argwhere(matrix[i,:] == 1)[0])
        if 1 in matrix[i,:]: 
            array[i] = np.argwhere(matrix[i,:] == 1)[0][0]
        else:
            array[i] = -1
    
    return(array)

def create_simplified_time_note_array(path='/Users/sorenh/documents/MIDI/HarryPotter.mid', v=True, save_file=True):
    mid = mido.MidiFile(path)
    array = []
        
    for i, track in enumerate(mid.tracks):
        for msg in track:
            m = msg.dict()
            if m['type'] == 'note_on':
                array.append(m['note'])
    array = np.array(array)
    if(save_file):
        filename = basename(path).split('.')[0] + '_array.txt'
        np.savetxt(filename,array,fmt='%i')
    return np.array(array)

def get_meta_messages(path='/Users/sorenh/documents/MIDI/HarryPotter.mid', v=True,):
    mid = mido.MidiFile('/Users/sorenh/documents/MIDI/HarryPotter.mid')
    messages = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if (msg.dict()['type'] == 'set_tempo'):
                messages.append(msg)
            if (msg.dict()['type'] == 'key_signature'):
                messages.append(msg)
            if (msg.dict()['type'] == 'program_change'):
                messages.append(msg)
            if (msg.dict()['type'] == 'control_change'):
                messages.append(msg)
            if (msg.dict()['type'] == 'time_signature'):
                messages.append(msg)
    
                
    return messages
    

def get_music_messages(path='/Users/sorenh/documents/MIDI/HarryPotter.mid', v=True):
    mid = mido.MidiFile(path)
    messages = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if (msg.dict()['type'] == 'note_on'):
                messages.append(msg)
            if (msg.dict()['type'] == 'note_off'):
                messages.append(msg)
                
    return messages

def play(meta, messages):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for m in meta:
        track.append(m)
    for m in messages:
        track.append(m)
    
    outport = mido.open_output('IAC Driver Bus 1')
    for msg in mid.play():
        outport.send(msg)

def play_midi(path='/Users/sorenh/documents/MIDI/HarryPotter.mid'):
    mid = mido.MidiFile(path)
    outport = mido.open_output('IAC Driver Bus 1')
    for msg in mid.play():
        outport.send(msg)
        
        

def split_motifs(messages, notes_per_motif=5):
    motif_list = []
#    print(type(motif_list))
    
    n = len(messages)
    i = 0

    while (i + 2*notes_per_motif < n):
#        print(type(motif_list))
        motif_list.append(messages[i:i + 2*notes_per_motif]) #because there are two messages per note
        i += 2*notes_per_motif
        
    motif_list.append(messages[i:])
    
    return motif_list
        
        
def stich_motifs(motifs, order):
    music = []
    
    for motif_number in order:
        for note in motifs[motif_number]:
            music.append(note)
            
    return music

# returns arrays with [note, velocity, length, time] for each note in a MIDI file
def get_features(path='/Users/sorenh/documents/MIDI/HarryPotter.mid', save_to_file=False):
    mid = mido.MidiFile(path)
    features = []
    incomplete_features = []
    time = 0
    
#    note_n = 0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            msgd = msg.dict()
            time += msgd['time']
            if (msgd['type'] == 'note_on'):
#                print(note_n, msgd['note'])
                incomplete_features.append(
                        [msgd['note'],msgd['velocity'],0,time])
            if (msgd['type'] == 'note_off'):
#                print(msg)
                for j in range(len(incomplete_features)):
                    if incomplete_features[j][0] == msgd['note']:
                        incomplete_features[j][2] = time - incomplete_features[j][3]
                        features.append(incomplete_features.pop(j))
                        break
    features = np.array(sorted(features,key=lambda x: x[3])) # makes sure that the events are ordered by the key pressed
#    features = np.array(features)
    if(save_to_file):
        filename = basename(path).split('.')[0] + '_features.txt'
        np.savetxt(filename,features,fmt='%i')                
    
    return features
    
def get_XY(features, n_notes = 5, save_to_file=False, name='harry_potter_features'):
    X = []
    Y = []
#    shape = features.shape
#    print(n)X
    i = 0
    while (i + n_notes + 1 < len(features) ):
        x = np.copy(features[i:i + n_notes])
        begin_time = x[0,3]
        x[:,3] = x[:,3] - begin_time
        X.append(x.flatten())
        y = np.copy(features[i + n_notes + 1])
        y[3] = y[3] - begin_time
        Y.append(y)
        i += 1
         
    X = np.array(X)
    Y = np.array(Y)
    
    if (save_to_file):
        np.savetxt(name + '_X.txt',X,fmt='%i')
        np.savetxt(name + '_Y.txt',Y,fmt='%i')
    
    return (X,Y)
        
def get_notes(path='/Users/sorenh/documents/MIDI/HarryPotter.mid'):
    mid = mido.MidiFile(path)
    notes = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            msgd = msg.dict()
            if (msgd['type'] == 'note_on'):
                notes.append( msgd['note'])    
    return notes

def get_note_training_set(prev_notes=10):
    midi_files = glob.glob("./Soren/songs/*.mid")
    X = []
    Y = []
    for file in midi_files:
        notes = get_notes(path=file)
        i = 0
        while(i + prev_notes + 1 < len(notes)):
            X.append(notes[i:i + prev_notes])
            Y.append(notes[i + prev_notes + 1])
            i += 1
    X = np.array(X)
    Y = np.array(Y)
    
    np.savetxt(str(prev_notes) + 'X.txt',X,fmt='%i')
    np.savetxt(str(prev_notes) + 'Y.txt',Y,fmt='%i')
    
    
    return (X,Y)

def create_midi_from_notes(notes, off=100, on=300):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=12, time=0))
    
    for note in notes:
        note = int(note)
        track.append(mido.Message('note_on', note=note, velocity=off, time=0))
        track.append(mido.Message('note_off', note=note, velocity=on, time=200))

    mid.save('generated_song.mid')
        
    
        
    
    

    

