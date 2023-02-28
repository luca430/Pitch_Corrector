import numpy as np
import scipy as sp
import librosa
from scipy.fft import fft, fftfreq
import soundfile as sf
from func import *
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')

# Parameters
scale = ["A","B","C#","D","E","F#","G#"] # scale to use for correction
alpha = 0.01 # correction tolerance
min_change = 9.5 # threshold of the derivative of pitch for audio section detection

# data import
data, sample_rate = librosa.load('Prova_Audio.wav',sr=44100)

# 1) PITCH DETECTION
print("Pitch detection...")
T = 1/sample_rate #period of one sampling
num = int(0.1*sample_rate) # 100ms sliding window
if num % 2 != 0: num += 1
freq = fftfreq(num,T)

B = create_overlapping_blocks(data,num) # devide audio in (overlapping) windows of num ms
Bitch = np.zeros(len(B)) # array pitches for each row of B
FFTs = np.zeros(np.shape(B)) + 1j*np.zeros(np.shape(B)) # array of FFTs for each row of B
for i in range(len(B)):
    FFT = fft(B[i,:])
    FFTs[i,:] += FFT
    Bitch[i] = freq[np.argmax(np.abs(FFT[:len(FFT)//2]))]

Bitch = clean_pitch(Bitch) # eliminate undesired sounds and noise

# 2) FINDING AUDIO SECTIONS
print("Deviding into sections...")
indices, d_Bitch = find_audio_sections(Bitch,min_change) # find the istants where the pitch has a significant change

# devide Bitch according to audio sections
p_chunks = [Bitch[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
p_chunks.append(Bitch[indices[-1]:])
# devide B according to audio sections
B_chunks = [B[indices[i]:indices[i+1],:] for i in range(len(indices)-1)]
B_chunks.append(B[indices[-1]:,:])

# 3) AUTO PITCH CORRECTION
print("Pitch correction...")
new_Bitch = []
pieces = []
for i in range(len(p_chunks)):

    audio = add_overlapping_blocks(B_chunks[i]) # recreate i-th audio section
    chunk = np.copy(p_chunks[i])

    # computing mean freq of chunk (avoiding transitions that can spoil the average)
    if len(chunk) > 2:
        mean_fr = np.mean(np.delete(chunk[1:-1],np.where(chunk[1:-1] == 0)))
    elif len(chunk) == 2:
        mean_fr = chunk[1]
    else:
        mean_fr = chunk[0]

    correct_note = min(notes, key=lambda x:abs(x-mean_fr)) # find closest note to sang frequency for each chunk

    # correction applied only if distance is above threshold
    if abs(mean_fr - correct_note)/correct_note > alpha and len(chunk) > 2:
        p_chunks[i] = pitch_correction(p_chunks[i],mean_fr,correct_note) # pitch correction (for the plot)
        audio = audio_correction(audio,mean_fr,correct_note) # audio-section correction (actual audio correction)
    
    new_Bitch = np.append(new_Bitch,p_chunks[i])
    pieces.append(audio)

pieces = np.array(pieces,dtype=object)
correct_data = join_pieces(pieces,num=num,deg=2) # join overlapping audio windows in a smart way to mask the correction
sf.write('correction.wav',correct_data,sample_rate) # create corrected audio
print("File saved")

# plots
if False:
    def plot_freq(fr):
        C = fr
        for i in range(len(C)):
            if C[i] < 65 or C[i] > 880: C[i] = None
        return C

    time = np.linspace(0,T*len(data),len(Bitch))
    Bitch = plot_freq(Bitch)
    new_Bitch = plot_freq(new_Bitch)

    plt.figure(1)
    plt.plot(time,Bitch)
    plt.plot(time,new_Bitch,color='red',lw=1)
    for n in all_notes:
        plt.axhline(n,ls='--',color="grey",lw=0.5)
    for n in notes:
        plt.axhline(n,color="orange",lw=0.5)
        plt.axhspan(n-n*alpha,n+n*alpha,alpha=0.2,color="red")
    for i in indices:
        plt.axvline(i*T*num/2,lw=0.5,color="lightgrey")

    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.ylim(200,800)
    plt.yscale("log")

if False:
    time = np.linspace(0,T*len(data),len(Bitch))
    plt.figure(2)
    plt.plot(time[1:-1],d_Bitch)
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.ylim(-2000,2000)
    plt.axhline(min_change,ls="--",color="red",lw=0.5)
    plt.axhline(-min_change,ls="--",color="red",lw=0.5)
    for i in indices:
        plt.axvline(i*T*num/2,lw=0.5,color="lightgrey")

plt.show()
