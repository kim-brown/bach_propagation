from mido import MidiFile
import os
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

for file in midi_files:
    print("Processing file ", file)
    for i, track in enumerate(file.tracks):
        #print('Track {}: {}'.format(i, track.name))
        for msg in track:
            #print(msg)
            continue
