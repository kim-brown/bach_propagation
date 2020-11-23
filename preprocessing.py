from mido import MidiFile
from play import play_midi
import os
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
    for i, track in enumerate(file.tracks):
        max_tracks = max(max_tracks, len(file.tracks))
        #print('Track {}: {}'.format(i, track.name))
        for msg in track:
            #print(msg)
            continue




# unfinished attempt at building piano roll representations of these midi files:


# there are at most 5 meaningful tracks in these files, so the tokens
# in the piano roll rep will have at most 5 notes in them

f = midi_files[0] # look at the first midi file
num_tracks = len(f.tracks)
tracks = f.tracks[1:num_tracks] # drop the metadata track

# get the length in ticks of each track (should all be nearly the same)
track_lengths = [sum([m.time for m in track]) for track in tracks]
max_track_length = max(track_lengths)

ticks_per_beat = f.ticks_per_beat
ticks_per_eigth_note = ticks_per_beat / 8 # ticks per eighth note
samples = max_track_length / ticks_per_eigth_note # total number of piano roll samples needed

piano_roll = np.zeros((int(samples), 1))

# now go through tracks at ticks_per_eigth_note intervals and
# find all of the currently playing notes and concatenate them into
# a vector which goes in piano-roll at the corresponding time step

# we know that messages don't overlap in a track, so we don't actually need the
# start and stop messages, we just need to know the note value and the start time

track = tracks[0] # example track to process
simple_track = []
time = 0
for msg in track:
    if msg.type == 'note_on':
        continue
    elif msg.type == 'note_off':
        simple_track.append([time, msg.note]) # start time, pitch
        time += msg.time

# simple_track is now a track representation where each note is represented
# as [start time in ticks, pitch value]













