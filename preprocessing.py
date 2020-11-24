from mido import MidiFile
from mido import MetaMessage
from play import play_midi
import os
import music21
import numpy as np


# https://mido.readthedocs.io/en/latest/

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

    return midi
    # return sNew #music21.midi.translate.streamToMidiFile(sNew)


data_dir = 'data/Bach/Fugue'

midi_files = []

for file in os.listdir(data_dir):
    if file.endswith(".mid"):
        path = os.path.join(data_dir, file)
        # score = music21.converter.parse(path) #Error: 'badly formed midi string: missing leading MTrk'
        # key = score.analyze('key')
        # print(key)
        try:
            midi_files.append(MidiFile(path))
        except OSError:
            continue
        except EOFError:
            continue

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


def sample_midi_track(track, interval, num_samples):
    """
    Samples from a single midi track, used for creating piano roll representation

    :param track: the midi track to sample from
    :param interval: the length of the interval in ticks used for sampling
    :param num_samples: the total number of samples for this track
    :return: a list of notes representing samples in the midi file taken every
    interval ticks
    """

    samples = np.zeros((num_samples,))
    sample_time = 0

    msg_index = 1  # index of the first "on" message
    note_start_time = track[msg_index].time
    note_duration = track[msg_index + 1].time - note_start_time

    while msg_index < len(track) - 3:

        while sample_time < note_start_time:
            samples[int(sample_time / interval)] = None  # there is no note playing
            sample_time += interval

        while sample_time < note_start_time + note_duration:
            if note_start_time <= sample_time < note_start_time + note_duration:
                samples[int(sample_time / interval)] = track[msg_index].note
                sample_time += interval

        msg_index += 2  # go to the next note ON message
        note_start_time += note_duration + track[msg_index].time
        note_duration = track[msg_index+1].time

    return samples


def piano_roll(midi_file):
    """
    This should build the piano roll representation of the midi file
    :param midi_file:
    :return:
    """
    num_tracks = len(midi_file.tracks)
    tracks = midi_file.tracks[1:num_tracks]  # drop the metadata track

    ticks_per_beat = midi_file.ticks_per_beat
    ticks_per_eighth_note = ticks_per_beat / 8  # ticks per eighth note

    # all tracks should have the same length
    track_length = sum([m.time for m in tracks[0]])
    num_samples = int(track_length / ticks_per_eighth_note)

    piano_roll = []
    for t in tracks:
        sampled_track = sample_midi_track(t, ticks_per_eighth_note, num_samples)
        piano_roll.append(sampled_track)

    # TODO concatenate the samples from all tracks into a single list of tokens where each
    #  token represents all notes being played at the current tick

    # ie somehow flatten the first dimension of piano_roll by combining note values

    piano_roll = np.stack(piano_roll)
    assert(np.array_equal(np.shape(piano_roll), (num_tracks-1, num_samples)))
    return piano_roll


pr = piano_roll(midi_files[0])
