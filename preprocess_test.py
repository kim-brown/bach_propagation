from preprocessing import get_files, normalize, piano_roll
from mido import MidiFile, MidiTrack, Message
from play import play_midi
import os



def piano_roll_to_midi(piano_roll, interval):
    """
    Converts the piano roll representation of a song back into a midi file
    for testing preprocessing and listening to network outputs

    :param piano_roll: the piano roll representation of a song
    :param interval: the time interval used for sampling when building the piano roll
    :return: a MidiFile object containing 3 tracks with messages based on the input piano roll
    """

    mid = MidiFile()
    track1 = MidiTrack()
    track2 = MidiTrack()
    track3 = MidiTrack()
    mid.tracks.append(track1)
    mid.tracks.append(track2)
    mid.tracks.append(track3)

    delta1, delta2, delta3 = 0, 0, 0
    for vector in piano_roll:
        vector = vector.split('-')

        if vector[0] == '0':
            delta1 += interval
        else:
            track1.append(Message('note_on', note=int(vector[0]), velocity=60, time=delta1))
            track1.append(Message('note_off', note=int(vector[0]), velocity=60, time=interval))
            delta1 = 0

        if len(vector) < 2 or vector[1] == '0':
            delta2 += interval
        else:
            track2.append(Message('note_on', note=int(vector[1]), velocity=60, time=delta2))
            track2.append(Message('note_off', note=int(vector[1]), velocity=60, time=interval))
            delta2 = 0

        if len(vector) < 3 or vector[2] == '0':
            delta3 += interval
        else:
            track3.append(Message('note_on', note=int(vector[2]), velocity=60, time=delta3))
            track3.append(Message('note_off', note=int(vector[2]), velocity=60, time=interval))
            delta3 = 0
    
    return mid


if __name__ == "__main__":
    # get a midi file from the data dir

    original_path = "./data/Bach/aof/cnt2.mid"
    midi_file = MidiFile(original_path)

    # tests normalize the tempo and key
    midi_file = normalize(midi_file)

    # create piano roll representation from the midi file
    piano_roll = piano_roll(midi_file)

    # convert piano roll back into a MidiFile object with same meta messages as original
    ticks_per_eighth_note = midi_file.ticks_per_beat / 2
    midi = piano_roll_to_midi(piano_roll, int(ticks_per_eighth_note))
    midi.ticks_per_beat = midi_file.ticks_per_beat
    midi.tracks.append(midi_file.tracks[0])

    # save to a file and play
    rel_path = "preprocess_test.mid"
    midi.save(rel_path)

    with open(rel_path) as song_file:
        print("playing processed")
        play_midi(song_file)

    # delete the saved midi file
    os.remove(rel_path)


    # play the original (before preprocessing):
    with open(original_path) as song_file:
        print("playing original")
        play_midi(song_file)






