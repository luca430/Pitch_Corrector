# functions and constants for the pitch corrector script

import numpy as np
import scipy as sp
import librosa
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf

# note frequencies
all_notes = np.array([32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,51.91,55.00,58.27,61.74,\
        65.41,69.30,73.42,77.78,82.41,87.31,92.50,98.00,103.83,110.00,116.54,123.47,\
        130.81,138.59,146.83,155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,\
        261.63,277.18,293.66,311.13,329.63,349.23,369.99,392.00,415.30,440.00,466.16,493.88,\
        523.25,554.37,587.33,622.25,659.25,698.46,739.99,783.99,830.61,880.00,932.33,987.77])

# useful functions
def custom_scale(scale):
    # allows to define a custom scale to follow for the correction

    indices = np.array([12*i for i in range(int(len(all_notes)/12))],dtype="int")
    notes = []

    for note in scale:
        if note == "C": notes.append(all_notes[indices])
        elif note == "C#" or note =="Db": notes.append(all_notes[indices+1])
        elif note == "D": notes.append(all_notes[indices+2])
        elif note == "D#" or note == "Eb": notes.append(all_notes[indices+3])
        elif note == "E": notes.append(all_notes[indices+4])
        elif note == "F": notes.append(all_notes[indices+5])
        elif note == "F#" or note == "Gb": notes.append(all_notes[indices+6])
        elif note == "G": notes.append(all_notes[indices+7])
        elif note == "G#" or "Ab": notes.append(all_notes[indices+8])
        elif note == "A": notes.append(all_notes[indices+9])
        elif note == "A#" or note == "Bb": notes.append(all_notes[indices+10])
        elif note == "B": notes.append(all_notes[indices+11])

    return np.unique(notes)

def create_overlapping_blocks(x,nw):
    # devide the input audio into overlapping windows of the same length

    x = np.insert(x,0,np.zeros(nw//2))
    x = np.append(x,np.zeros(nw//2))

    step = nw//2
    nb = int((len(x) - nw)/step) + 1 # total steps

    B = np.zeros((nb, nw)) # matrix of windows

    for i in range(nb):
        offset = i*step
        B[i,:] = x[offset:nw+offset]

    return B

def add_overlapping_blocks(B):
    # join subsequent overlapping audio windows of the same length

    nb, nw = np.shape(B)
    step = nw//2

    n = (nb + 1)*step

    x = np.zeros(n)

    for i in range(nb-1):
        x[i*step:(i+1)*step] = B[i,:step]
    x[-nw:] = B[-1,:]

    return x

def join_pieces(pieces,num=4410,deg=1):
    # join subsequent overlapping audio sections of different lengths

    n = num//2

    a = 1/n**(deg)
    win = [a*x**(deg) for x in range(n)]
    win = np.append(win,np.flip(win))
    win[:n] = [1 - w for w in win[n:]]

    pieces[0][-n:] *= win[n:]
    out = pieces[0]
    out[-n:] += pieces[1][:n]*win[:n]
    for i in range(1,len(pieces)-1):
        out = np.append(out,pieces[i][n:])
        out[-n:] *= win[n:]
        out[-n:] += pieces[i+1][:n]*win[:n]
    out = np.append(out,pieces[-1][n:])

    return out[n:-n]

def pitch_correction(fs,mean,correct):
    # shift the average pitch of the input to the closest correct note

    new_fs = []
    if mean != 0:
        for f in fs:
            new_fs.append(f - (mean - correct))
        fs = new_fs

    return fs

def audio_correction(audio,mean,correct):
    # change the pitch of the audio section

    rate = mean/correct
    audio1 = np.flip(audio[:len(audio)//2])
    audio2 = audio[len(audio)//2:]
    audio1 = librosa.effects.time_stretch(audio1,rate=rate)
    audio2 = librosa.effects.time_stretch(audio2,rate=rate)
    new_audio = np.append(np.flip(audio1),audio2)

    new_audio = sp.signal.resample(new_audio,len(audio))
    audio = new_audio

    return audio

def clean_pitch(pitch):
    # clean input avoiding noise

    for i in range(1,len(pitch)-1):
    
        # cancel out-of-range frequencies
        if pitch[i-1] < 32 or pitch[i-1] > 988: pitch[i-1] = np.nan

        # cancel strange spikes
        if not np.isnan(pitch[i-1]):
            ratio12 = pitch[i-1]/pitch[i]
            ratio32 = pitch[i+1]/pitch[i]
            if pitch[i-1] == pitch[i+1] and pitch[i] != pitch[i-1]:
                if ratio12 < 0.8 or ratio12 > 1.2: pitch[i] = np.nan
            elif pitch[i-1] != pitch[i] and pitch[i] != pitch[i+1] and pitch[i-1] != pitch[i+1]:
                if ratio12 < 0.8 and ratio32 < 0.8: pitch[i] = np.nan
                elif ratio12 > 1.2 and ratio32 > 1.2: pitch[i] = np.nan
        
        # residual out-of-range frequencies
        if pitch[-2] < 32 or pitch[-2] > 988: pitch[-2] = np.nan
        if pitch[-1] < 32 or pitch[-1] > 988: pitch[-1] = np.nan

    return pitch

def find_audio_sections(pitch,thresh):
    # devide audio according to significant pitch changes

    d_pitch = []
    indices = [] # array of indeces of song sections

    for i in range(1,len(pitch)-1):
        d_pitch.append((pitch[i-1] - pitch[i+1])/2) # derivative of pitch

        # find discontinuities
        if np.isnan(pitch[i]) and not np.isnan(pitch[i+1]):
            indices.append(i+1)
        elif np.isnan(pitch[i+1]) and not np.isnan(pitch[i]):
            indices.append(i)

    # find pitch changes
    for i in range(1,len(pitch)-1,2):
        if abs(pitch[i-1] - pitch[i+1])/2 > thresh:
            indices.append(i)

    indices =  np.unique(indices)
    if indices[0] != 0: indices = np.insert(indices,0,0) # ensure that first index is 0

    return indices, d_pitch