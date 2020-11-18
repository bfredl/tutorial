import warnings
import nussl

signal = nussl.AudioSignal("/home/bjorn/ljudbild/lydmorean/anoraak/evolve_cut.wav")
print(signal)

st = signal.stft()
signal.stft_data

"""
%matplotlib tk
"""

signal1 = signal
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 9))
plt.subplot(311)
plt.title('Linear-frequency spectrogram')
nussl.utils.visualize_spectrogram(signal1)

plt.subplot(312)
plt.title('Log-frequency spectrogram')
nussl.utils.visualize_spectrogram(signal1, y_axis='log')

plt.subplot(313)
plt.title('Mel-frequency spectrogram')
nussl.utils.visualize_spectrogram(signal1, y_axis='mel')

plt.tight_layout()
#plt.show()

import numpy as np

lp_stft = signal1.stft_data.copy()
lp_cutoff = 1000  # Hz
frequency_vector = signal1.freq_vector  # a vector of frequency values for each FFT bin
idx = (np.abs(frequency_vector - lp_cutoff)).argmin()  # trick to find the index of the closest value to lp_cutoff
lp_stft[idx:, :, :] = 0.0j  # every freq above lp_cutoff is 0 now

signal1_lp = signal1.make_copy_with_stft_data(lp_stft)
signal1_lp.istft()
em = signal1_lp.embed_audio()
_.filename
from subprocess import call
call(["mpv", em.filename])

mix = signal
repet = nussl.separation.primitive.Repet(mix)

bg_mask, fg_mask = repet.run()
bg_mask_arr = bg_mask.mask
fg_mask_arr = fg_mask.mask

# Multiply the masks to the magnitude spectrogram
mix_mag_spec = mix.magnitude_spectrogram_data
bg_no_phase_spec = mix_mag_spec * bg_mask_arr
fg_no_phase_spec = mix_mag_spec * fg_mask_arr

# Make new AudioSignals for background and foreground without phase
bg_no_phase = mix.make_copy_with_stft_data(bg_no_phase_spec)
_ = bg_no_phase.istft()
fg_no_phase = mix.make_copy_with_stft_data(fg_no_phase_spec)
_ = fg_no_phase.istft()

bg_no_phase.embed_audio().filename
fg_no_phase.embed_audio().filename



def apply_mask_with_noisy_phase(mix_stft, mask):
    mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
    src_magnitude = mix_magnitude * mask
    src_stft = src_magnitude * np.exp(1j * mix_phase)
    return src_stft

bg_stft = apply_mask_with_noisy_phase(mix.stft_data, bg_mask_arr)
fg_stft = apply_mask_with_noisy_phase(mix.stft_data, fg_mask_arr)


# Make new AudioSignals for background and foreground with phase
bg_phase = mix.make_copy_with_stft_data(bg_stft)
_ = bg_phase.istft()
fg_phase = mix.make_copy_with_stft_data(fg_stft)
_ = fg_phase.istft()

call(["clementine", bg_phase.embed_audio().filename])
call(["clementine", fg_phase.embed_audio().filename])
fg_phase.embed_audio().filename
