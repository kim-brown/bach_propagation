from mido import MidiFile, MetaMessage
from play import play_midi
import os
import music21
import numpy as np


# https://mido.readthedocs.io/en/latest/

train_test_split = 0.8

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
                #print(path)
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

    key = "C" #if there's no key
    
    for i, track in enumerate(midi_file.tracks):
        #max_tracks = max(max_tracks, len(midi_file.tracks))
        # print('Track {}: {}'.format(i, track.name))
        # I think this makes all the songs the same tempo? not sure
        # TODO: normalize midi_file so the tempos the same
        midi_file.ticks_per_beat = 120
        # TODO: normalize midi_file so it's in C
        
        
        for msg in track: 
            if isinstance(msg, MetaMessage):
                if msg.type == 'key_signature':
                    key = msg.key #determines key of song
                    print(key)
        reference = {"B#": 0, "C":0, "C#": -1, "Db": -1, "D":-2, "D#":-3, "Eb":-3, 
        "E":-4, "Fb": -4, "E#":-5, "F":-5, "F#":-6, "Gb":-6, "G":5, "G#":4, "Ab":4,
        "A":3, "A#": 2, "Bb":2, "B":1, "Cb":1,
        "B#m": -3, "Cm":-3, "C#m": -4, "Dbm": -4, "Dm":-5, "D#m":-6, "Ebm":-6, 
        "Em":-7, "Fbm": -7, "E#m":4, "Fm":4, "F#m":3, "Gbm":3, "Gm":2, "G#m":1, "Abm":1,
        "Am":0, "A#m": -1, "Bbm":-1, "Bm":-2, "Cbm":-2} 
        #dictionary saying how many steps should we be transposing, major pieces are normalized to C, minor to A
        difference = reference[key]
        print("difference:", difference)
        for msg in track:
            if msg.type == "note_on":
                    msg.note += difference
            continue


    return midi_file
    # return sNew #music21.midi.translate.streamToMidiFile(sNew)


"""max_tracks = 0
for file in get_files('data/bach/aof'):
    print("Processing file ", file)
    normalize(file)
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
            continue"""



def sample_midi_track(track, interval, num_samples):
    """
    Samples from a single midi track, used for creating piano roll representation

    :param track: the midi track to sample from
    :param interval: the length of the interval in ticks used for sampling
    :param num_samples: the total number of samples for this track
    :return: a list of notes representing samples in the midi file taken every
    interval ticks
    """

    # this offsets our sampling so that it occurs not exactly on every eighth
    # note, but slightly after (shifted by 1/4 of an eighth note). the notes in
    # the midi files aren't quantized, so doing this ensures the model doesn't
    # miss notes that occur ever-so-slightly after the beat.
    cur_time = interval // 4
    
    next_msg_time = 0
    arr_index = 0
    samples = np.zeros(num_samples, dtype=np.int32)
    
    for msg in track:
        #print(msg)
        next_msg_time += msg.time
        while cur_time < next_msg_time:
            # if msg.type=='note_off', add the note (since this means it's
            # currently on and will be turned off at next_msg_time).
            # otherwise, it remains 0 (i.e., the note is off)
            if msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                samples[arr_index] = msg.note
            cur_time += interval
            arr_index += 1
        
    return samples


def piano_roll(midi_file):
    """
    This should build the piano roll representation of the midi file
    :param midi_file:
    :return:
    """
    #midi_file = normalize(midi_file) #new! may break things, but supposed to get everythign in C an = 120
    tracks = midi_file.tracks[1:]  # drop the metadata track
    # it seems like midi always treats a quarter note as a beat, regardless of time signature
    ticks_per_eighth_note = midi_file.ticks_per_beat / 2
    # print(ticks_per_eighth_note)
    
    # if there are more than 3 tracks, randomly choose 3 of them to work with
    if len(tracks) > 3:
        tracks = np.random.choice(tracks, 3, replace=False)
    
    # the tracks seem to be slightly different lengths for some reason, so take
    # the max length to determine the number of samples.
    
    # see comments in sample_midi_track for why we add ticks_per_eighth_note // 4
    max_length = max([sum([m.time for m in track]) for track in tracks]) + ticks_per_eighth_note // 4
    num_samples = np.ceil(max_length / ticks_per_eighth_note).astype('int')

    piano_roll = np.array([sample_midi_track(track, ticks_per_eighth_note, num_samples) for track in tracks])
    piano_roll = np.transpose(piano_roll)
    
    # sort the notes at each timestep from lowest to highest and concatenate them into a single string
    piano_roll = np.sort(piano_roll)
    piano_roll = np.array(["-".join([str(note) for note in notes]) for notes in piano_roll])

    #print(piano_roll[:50])
    return piano_roll


def get_batch(inputs, labels, start, batch_size):
    """
    Batch the inputs and labels.
    :param inputs: a list or numpy array
    :param labels: a list or numpy array of the same length
    :param start: index at which to begin the batch
    :param batch_size: size of the batch
    :return: batched_inputs, batched_labels
    """
    return inputs[start:start+batch_size], labels[start:start+batch_size]


def build_vocab(tokens):
    """
    Create a dictionary which maps different tokens to ids
    :params: tokens (string representation of notes playing at a timestep)
    :return: token_to_id: dictionary from token -> token_id
    """
    token_to_id = {}
    highest_id = 0
    for t in tokens:
        if t not in token_to_id:
            token_to_id[t] = highest_id
            highest_id += 1
    return token_to_id

def tokens_to_ids(tokens, token_to_id):
    ids = []
    for t in tokens:
        if t in token_to_id:
            ids.append(token_to_id[t])
        else:
            ids.append(0) # not sure if there are any consequences to this
    return ids


def get_data():
    """
    Combine all piano rolls into one array, divide into training + testing data, and inputs + labels.
    :return: train_inputs, train_labels, test_inputs, test_labels
    """

    midi_files = []

    pieces = ['aof', 'brandenb', 'cantatas', 'cellosui', 'chorales', 'fugues', 'gold', 'invent',
              'organ', 'organcho', 'partitas', 'sinfon', 'suites', 'wtcbki', 'wtcbkii']

    for p in pieces:
        data_dir = 'data/bach/' + p
        for midi_file in get_files(data_dir):
            midi_files.append(midi_file)

    print(len(midi_files), " Midi Files processed for training")

    if len(midi_files) > 0:
        inputs = piano_roll(midi_files[0])
    rolled = []
    for f in range(1, len(midi_files)):
        roll = piano_roll(midi_files[f])
        rolled.append(roll)
    all_notes = np.concatenate(rolled)
    train_length = int(train_test_split * len(all_notes))
    train_inputs = all_notes[:train_length-1]
    token_to_id = build_vocab(train_inputs)
    train_input_ids = tokens_to_ids(train_inputs, token_to_id)
    train_label_ids = tokens_to_ids(all_notes[1:train_length], token_to_id)
    test_input_ids = tokens_to_ids(all_notes[train_length:-1], token_to_id)
    test_label_ids = tokens_to_ids(all_notes[train_length+1:], token_to_id)
    return train_input_ids, train_label_ids, test_input_ids, test_label_ids, token_to_id


if __name__ == "__main__":
    train_input_ids, train_label_ids, test_input_ids, test_label_ids, token_to_id = get_data()

    print(np.shape(train_input_ids))
    print(np.shape(train_label_ids))
    print(np.shape(test_input_ids))
    print(np.shape(test_input_ids))
