from mido import MidiFile
from play import play_midi
import os
import music21
import numpy as np
# https://mido.readthedocs.io/en/latest/

data_dir = 'data/Bach/Fugue'

midi_files = []

for file in os.listdir(data_dir):
    if file.endswith(".mid"):
        path = os.path.join(data_dir, file)
        try:
            midi_files.append(MidiFile(path))
        except OSError:
            continue
        except EOFError:
            continue

max_tracks = 0
for file in midi_files:
    print("Processing file ", file)
    print(normalize(file))
    #score = music21.converter.parse("data/Bach/Fugue/"+file)
    #k = score.analyze('key')
    #print("file key", k)
    for i, track in enumerate(file.tracks):
        max_tracks = max(max_tracks, len(file.tracks))
        #print('Track {}: {}'.format(i, track.name))
        for msg in track:
            #print(msg)
            continue

def normalize(midi_file):
    """
    this should take a single midi track and normalize it so that it is in C and the tempos
    are somewhat in line with the other ones

    :param midi_file: midi track to change
    :return: the midi track with the sounds normalized
    """

    #TODO: normalize midi_file so it's in C
    s = music21.midi.translate.midiFileToStream(midi_file) #not working for some reason... 
    #"MidiFile Object has no attribute 'ticksPerQuarterNote'"
    k = s.analyze('key')
    i = interval.Interval(k.tonic, pitch.Pitch('C'))
    sNew = s.transpose(i)
    #TODO: normalize midi_file so the tempos the same
    return sNew


def sample_midi_track(track, interval):
    """
    Samples from a single midi track, used for creating piano roll representation

    :param track: the midi track to sample from
    :param interval: the length of the interval in ticks used for sampling
    :return: a list of notes representing samples in the midi file taken every
    interval ticks
    """

    time = 0
    track_length = sum([m.time for m in track])
    msg_index = 1 # message 0 is metadata
    msg = track[msg_index]
    samples = []
    next_sample_time = 30

    while next_sample_time < track_length:
        while time < next_sample_time:
            msg_index += 1
            if msg_index >= len(track):
                return np.array(samples, dtype=np.int32)
            msg = track[msg_index]
            time += msg.time
        samples.append(msg.note)
        next_sample_time += interval

    return np.array(samples, dtype=np.int32)

def piano_roll(midi_file):
    """
    This should build the piano roll representation of the midi file
    :param midi_file:
    :return:
    """
    num_tracks = len(midi_file.tracks)
    tracks = midi_file.tracks[1:num_tracks]  # drop the metadata track

    ticks_per_beat = midi_file.ticks_per_beat
    ticks_per_eigth_note = ticks_per_beat / 8  # ticks per eighth note

    for t in tracks:
        # all tracks should have the same length
        track_length = sum([m.time for m in track])
        print("track total length in ticks: ", track_length)

        samples = sample_midi_track(t, ticks_per_eigth_note)

        # we want this so that all the tracks are the same length and we
        # can concatenate into a single list of vectors
        print(np.shape(samples))

    return np.array([], dtype=np.int32)

    # TODO test/debug sample_midi_file() so that it works correctly and all tracks from the
    #  same midi file have the same number of samples

    # TODO concatenate all samples of tracks from a song into a single list of tokens where each
    #  token represents all notes being played at the current tick




pr = piano_roll(midi_files[0])














