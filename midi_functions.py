#!/usr/bin/env python3
#from mido import MidiFile
import mido
from os.path import basename
import numpy as np

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
    mid = mido.MidiFile('/Users/sorenh/documents/MIDI/HarryPotter.mid')
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

def split_motifs(messages, notes_per_motif=5):
    motif_list = []
#    print(type(motif_list))
    
    n = len(messages)
    i = 0

    while (i + notes_per_motif < n):
#        print(type(motif_list))
        motif_list.append(messages[i:i+notes_per_motif])
        i += notes_per_motif
        
    motif_list.append(messages[i:])
    
    return motif_list
        
        
def stich_motifs(motifs, order):
    music = []
    
    for motif_number in order:
        for note in motifs[motif_number]:
            music.append(note)
            
    return music
    
    
    

    

