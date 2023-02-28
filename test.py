import numpy as np
import scipy as sp
import librosa
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf
from func import *
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')

scale = ["C","D","E","F","G"]
s = custom_scale(scale)
print(all_notes)

