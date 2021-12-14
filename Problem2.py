import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from IPython.display import Audio

fileName1 = 'audio1.wav'
fs1, data1 = read(fileName1)
print('Sampling frequency of ', fileName1, ': ', fs1, sep = '')
print('Length data: ', len(data1), sep = '')

plt.plot(data1)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Waveform of ' + fileName1)
plt.show()
Audio(data1, rate = fs1)