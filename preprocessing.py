from mido import MidiFile
from mido import MetaMessage
from play import play_midi
import os
import music21
import numpy as np


# https://mido.readthedocs.io/en/latest/

def get_files(data_dir):
    """
    :param data_dir: string, name of directory containing midi files
    :return: a list of midi files in that directory
    """
    midi_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".mid"):
            path = os.path.join(data_dir, file)
            #score = music21.converter.parse(path) #Error: 'badly formed midi string: missing leading MTrk'
            #key = score.analyze('key')
            #print(key)
            try:
                midi_files.append(MidiFile(path))
                print(path)
            except OSError:
                continue
            except EOFError:
                continue
    return midi_files

def normalize(midi_file):
    """
    this should take a single midi track and normalize it so that it is in C and the tempos
    are somewhat in line with the other ones

    :param midi_file: midi track to change
    :return: the midi track with the sounds normalized
    """

    # TODO: normalize midi_file so it's in C
    # s = music21.midi.translate.midiFileToStream(midi_file) #need to convert to Stream to normalize
    # "MidiFile Object has no attribute 'ticksPerQuarterNote'"
    """k = s.analyze('key')
    i = interval.Interval(k.tonic, pitch.Pitch('C'))
    sNew = s.transpose(i)"""
    # TODO: normalize midi_file so the tempos the same -- done in for msg in track I think?
    # The meta message ‘set_tempo’ can be used to change tempo during a song.
    """tempo = int((60.0 / bpm) * 1000000)
    track.append(MetaMessage('set_tempo', tempo=tempo))"""

    return midi_file
    # return sNew #music21.midi.translate.streamToMidiFile(sNew)

"""
max_tracks = 0
for file in midi_files:
    print("Processing file ", file)
    # print(normalize(file))
    # score = music21.converter.parse("data/Bach/Fugue/"+file)
    # k = score.analyze('key')
    # print("file key", k)
    for i, track in enumerate(file.tracks):
        max_tracks = max(max_tracks, len(file.tracks))
        # print('Track {}: {}'.format(i, track.name))
        # I think this makes all the songs the same tempo? not sure
        for msg in track:
            if isinstance(msg, MetaMessage):
                if msg.type == 'set_tempo':
                    tempo = int((60.0 / 120) * 1000000)  # 60/bpm
                    msg = MetaMessage('set_tempo', tempo=tempo)
                print(msg)
            continue
"""


def sample_midi_track(track, interval, num_samples):
    """
    Samples from a single midi track, used for creating piano roll representation

    :param track: the midi track to sample from
    :param interval: the length of the interval in ticks used for sampling
    :param num_samples: the total number of samples for this track
    :return: a list of notes representing samples in the midi file taken every
    interval ticks
    """

    cur_time = 0
    next_msg_time = 0
    arr_index = 0
    samples = np.zeros(num_samples)
    msgs = track[1:] # exclude track[0], which is metadata
    
    for msg in msgs:
        next_msg_time += msg.time
        while cur_time < next_msg_time:
            # if msg.type=='note_off', add the note (since this means it's
            # currently on and will be turned off at next_msg_time).
            # otherwise, it remains 0 (i.e., the note is off)
            if msg.type=='note_off':
                samples[arr_index] = msg.note
            cur_time += interval
            arr_index += 1
        
    print(samples[len(samples)-20:])
    return samples


def piano_roll(midi_file):
    """
    This should build the piano roll representation of the midi file
    :param midi_file:
    :return:
    """
    tracks = midi_file.tracks[1:]  # drop the metadata track
    ticks_per_eighth_note = midi_file.ticks_per_beat / 2  # ticks per eighth note
    print(ticks_per_eighth_note)
    
    # TODO: we want no more than 3 tracks, so if there are more, randomly choose 3 of them to work with
    
    # the tracks seem to be slightly different lengths for some reason, so take
    # the max length to determine the number of samples
    for track in tracks:
        print(sum([m.time for m in track]))
    
    max_length = max([sum([m.time for m in track]) for track in tracks])
    num_samples = np.ceil(max_length / ticks_per_eighth_note).astype('int')

    piano_roll = np.array([sample_midi_track(track, ticks_per_eighth_note, num_samples) for track in tracks])
    
    # TODO concatenate the samples from all tracks into a single list of tokens where each
    #  token represents all notes being played at the current tick

    # ie somehow flatten the first dimension of piano_roll by combining note values

    print(piano_roll.shape)
    return piano_roll


midi_files = get_files('data/Bach/Fugue')
for midi_file in midi_files:
    piano_roll(midi_file)
