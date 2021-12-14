import librosa as lb
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


N_fft = 1024
# Hop length is 512.

def updataVarialbe(H, P, W, alpha):
    """
    Calculate delta.
    :param H:
    :param W:
    :param alpha:
    :return:
    """

    H_next = H.copy()
    H_prev = H.copy()

    P_up = P.copy()
    P_down = P.copy()

    # Shift H back one column.
    # Change the last column to zero-column.
    # Calculate H_h,(i + 1) matrix in (23).

    # Shift H go up one column.
    # Change the first column to zero-column.
    # Calculate H_h,(i - 1) matrix in (23).
    for i in range(len(H)):
        H_next[i] = np.roll(H_next[i], -1)
        H_next[i][-1] = 0

        H_prev[i] = np.roll(H_prev[i], 1)
        H_prev[i][0] = 0



    # Shift P go up one row.
    # Change the first column to zero-column.
    # Calculate P_h+1,i matrix in (23).
    P_up = np.roll(P_up, (-1) * len(P[0]))
    P_up[-1] = np.zeros(len(P[0]))

    # Shift P back one row.
    # Change the last column to zero-column.
    # Calculate P_h-1,i matrix in (23).
    P_down = np.roll(P_down, len(P[0]))
    P_down[0] = np.zeros(len(P[0]))

    """
    print('H_prev', H_prev)
    print()
    print('H', H)
    print()
    print('H_next', H_next)
    """

    # Calculate delta matrix in (23).
    delta = alpha * ((H_prev - 2 * H + H_next) / 4) - \
            (1 - alpha) * ((P_down - 2 * P + P_up) / 4)

    # print(delta)

    return delta


def plotSpectrogram(fileName, H_spec, P_spec, H, P, fs, K_max, alpha, gamma):
    """
    Plot four plot of spectrogram in one figure.
    :param H_spec: spectrogram H before binaried.
    :param P_spec: spectrogram P before binaried.
    :param H: spectrogram H after binaried.
    :param P: spectrogram P after binaried.
    :param fs: sampling frequency.
    """

    plt.figure(figsize=(10, 5))

    fig = plt.gcf()
    fig.canvas.set_window_title(f'Separation result of \'{fileName}\' with alpha='
                                f'{alpha} - gamma={gamma} - K_max={K_max}')

    plt.subplot(2, 2, 1)
    lb.display.specshow(lb.amplitude_to_db(np.abs(H_spec), ref=np.max), sr=fs, \
                        x_axis='time', y_axis='linear')

    plt.title('H spectrogram before binarized')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    lb.display.specshow(lb.amplitude_to_db(np.abs(P_spec), ref=np.max), sr=fs, \
                        x_axis='time', y_axis='linear')

    plt.title('P spectrogram before binarize')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    lb.display.specshow(lb.amplitude_to_db(np.abs(H), ref=np.max), sr=fs, \
                        x_axis='time', y_axis='linear')

    plt.title('H spectrogram after binarized')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    lb.display.specshow(lb.amplitude_to_db(np.abs(P), ref=np.max), sr=fs, \
                        x_axis='time', y_axis='linear')

    plt.title('P spectrogram after binarized')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def setInitialValue(data, gamma):
    """
    Calculate short Fourier transform of the
    input signal, the range-compressed version of the
    spectrogram. Set initial value of harmonic
    component and percusive component of the
    spectrogram.
    :param data is input data.
            gamma is used in calculate
            range-compress version of spectrogram.
    :return: STFT of input signal.
             W range-compressed version of
             the spectrogram.
             H, P the harmonic and percusive
             component matrix.

    """

    # Calculate STFT of input signal.
    # Step 1.
    F_stft = lb.stft(data, n_fft=N_fft, hop_length=int(N_fft/2))

    # Calculate range-compressed version
    # the power spectrogram.
    # Step 2 - equation (24).
    W = np.abs(F_stft) ** (2 * gamma)

    # Set initial value.
    # Step 3 - equation (25).
    H = 0.5 * W
    P = 0.5 * W

    return F_stft, W, H, P


def process(H, P, W, alpha):

    # Calculate update variable delta in (23).
    # Step 4.
    delta = updataVarialbe(H, P, W, alpha)

    # Updata H_h,i and P_h,i in (26), (27).
    # Step 5.

    for i in range(len(H)):
        for j in range(len(H[i])):
            H[i][j] = min(max(H[i][j] + delta[i][j], 0), W[i][j])

    P = W - H

    return H, P


def binarizeSeperationResult(H, P, W):
    """
    Step 7 - Equation (28).
    :param H:
    :param P:
    :param W:
    :return: (H_h,i, P_h,i) = (0, W_h,i) if H_h,i < P_h,i
             (H_h,i, P_h,i) = (W_h,i, 0) if H_h,i >= P_h,i
    """

    for i in range(len(H)):
        for j in range(len(H[i])):
            H[i][j] = W[i][j] - (W[i][j] * int(H[i][j] < P[i][j]))
            P[i][j] = W[i][j] - H[i][j]

    return H, P


def convertIntoWaveform(H, P, F, LENGTH, gamma):
    """
    Convert H, P into waveforms.
    Step 8 - Equation (29), (30).
    :param H:
    :param P:
    :param F:
    :return:
    """

    arg_F_with_j = np.angle(F) * 1j

    H_complex = (H ** (1/(2*gamma))) * np.exp(arg_F_with_j)
    P_complex = (P ** (1/(2*gamma))) * np.exp(arg_F_with_j)

    h = lb.istft(H_complex, hop_length=int(N_fft/2), length=LENGTH)
    p = lb.istft(P_complex, hop_length=int(N_fft/2), length=LENGTH)

    return h, p


def writeToFile(fileName, h, p, fs, alpha, gamma):
    """
    Write two audio file of the harmonic
    and percussive component of the origninal
    file with name has information of paramter
    alpha and gamma.
    :param h: Waveforms of harmonic component.
    :param p: Waveforms of percussive component.
    :param fs: Sampling frequency.
    :param alpha:   Control parameter.
    :param gamma:   Compress parameter.
    """

    write(fileName[:-4] + '_Harmonic_a_' + str(alpha * 10) + \
          '_g_' + str(gamma * 10) + '.wav', fs, h)
    write(fileName[:-4] + '_Percussive_a_' + str(alpha * 10) + \
          '_g_' + str(gamma * 10) + '.wav', fs, p)


def calculateSNR(data, p):
    """
    Calculate the signal-to-noise ratio.
    :param data: original data.
    :param p: percusive component wave form.
    :return: signal-to-noise ratio in dB.
    """

    numerator = data ** 2
    denominator = (data - p) ** 2

    snr = 10 * np.log10(sum(numerator) / sum(denominator))

    return snr


def main():

    # Get fileName and input parameters.
    fileName = input('Enter the song file name with extension (.wav): ')
    K_max = int(input('Enter the number of interation: '))
    alpha = float(input('Enter alpha: '))
    gamma = float(input('Enter gamma: '))

    # Load file.
    data, fs = lb.load(fileName)

    # F_stft is the short Fourier transfomr of the input signal.
    # W is range-compresed version of the power spectrogram.
    # H is the matrix of harmonic compnonet H_h,i, h is row
    # i is the column.
    # P is the matrix of percussive component P_h,i h is row
    # i is the column.
    F_stft, W, H, P = setInitialValue(data, gamma)

    # Step 6.
    k = 0
    while k < K_max:

        # Printout loop counters.
        print(k)

        # Step 4 and step 5.
        H, P = process(H, P, W, alpha)

        # Update loop counter.
        k += 1

    print('Done.')

    # Store the spectrogram before binarized.
    H_spec = H.copy()
    P_spec = P.copy()

    # Step 7.
    H, P = binarizeSeperationResult(H, P, W)

    # Step 8.
    h, p = convertIntoWaveform(H, P, F_stft, len(data), gamma)

    # Calculate signal-to-noise ratio
    snr = calculateSNR(data, p)
    print('\nSignal-to-noise rate:', snr, '(dB)')

    # Write two the separation of the harmonic
    # and the percussive into two files.
    writeToFile(fileName, h, p, fs, alpha, gamma)

    # Plot the results.
    plotSpectrogram(fileName, H_spec, P_spec, H, P, fs, K_max, alpha, gamma)


main()


