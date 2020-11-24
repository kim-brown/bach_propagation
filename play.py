import music21

import pygame
import os

# below functions are taken from here: https://github.com/sarthak15169/Deep-Music/blob/master/play.py
# just to play around with the midi files, this is not our code!!

def play_music(music_file):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print ("Music file %s loaded!" % music_file)
    except pygame.error:
        print ("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)
# pick a midi music file you have ...
# (if not in working folder use full path)

def play_midi(midi_file):
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024    # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)
    try:
        play_music(midi_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

if __name__ == "__main__":

    script_dir = os.path.dirname(__file__)
    rel_path = "data/Bach/Fugue/Fugue20.mid"
    abs_file_path = os.path.join(script_dir, rel_path)

    with open(abs_file_path) as song_file:
        
        play_midi(song_file)
        #based on listening for 30 sec may not be accurate, i'm not good at flat keys
        #Fugue1 is C Major
        #Fugue2 is C minor
        #Fugue3 is Db Major?
        #Fugue4 is Ab Major?
        #Fugue5 is D Major
        #Fugue6 is D minor
        #Fugue7 is Bb Major
        #Fugue8 is weird but also i lowkey dig it Eb?
        #Fugue9 is E minor
        #Fugue10 is E major
        #Fugue11 is F minor?
        #Fugue12 is 12 tone :o C minor if i have to guess
        #Fugue13 is Db Major? 
        #Fugue14 is F# minor?
        #Fugue15 is G major
        #Fugue16 is Eb major
        #Fugue17 is Ab Major
        #Fugue18 is Ab Major
        #Fugue19 is A Major
        #Fugue20 is A minor
