import warnings
import nussl
import numpy as np
import torch
torch.cuda.get_device_name()
dev = torch.device('cuda')
torch.randn(3,3).to(dev)

def lyssna(sig):
    fil = sig.embed_audio().filename
    call(["audacious", fil])
    print(fil)
    return fil


signal = nussl.AudioSignal("/home/bjorn/ljudbild/lydmorean/anoraak/evolve_cut.wav")
print(signal)

st = signal.stft()
mix = signal
##
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

repet = nussl.separation.primitive.Repet(mix)
repet = nussl.separation.primitive.RepetSim(mix)

bg_mask, fg_mask = repet.run()
bg_mask_arr = bg_mask.mask
fg_mask_arr = fg_mask.mask



# Multiply the masks to the magnitude spectrogram
mix_mag_spec = mix.magnitude_spectrogram_data
bg_no_phase_spec = mix_mag_spec * bg_mask_arr
fg_no_phase_spec = mix_mag_spec * fg_mask_arr

# Make new AudioSignals for background and foreground without phase
_ = bg_no_phase.istft()
fg_no_phase = mix.make_copy_with_stft_data(fg_no_phase_spec)
_ = fg_no_phase.istft()

bg_no_phase.embed_audio().filename
fg_no_phase.embed_audio().filename

def apply_mask(mix, mask, phase=True):
    mix_mag_spec = mix.magnitude_spectrogram_data
    if phase:
        mix_stft = mix.stft_data
        mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
        src_magnitude = mix_magnitude * mask
        stft_spec = src_magnitude * np.exp(1j * mix_phase)
    else:
        stft_spec = mix_mag_spec * mask
    result = mix.make_copy_with_stft_data(stft_spec)
    result.istft()
    return result

bg_no_phase = apply_mask(mix, bg_mask.mask, phase=False)
fg_no_phase = apply_mask(mix, fg_mask.mask, phase=False)
bg_phase = apply_mask(mix, bg_mask.mask, phase=True)
fg_phase = apply_mask(mix, fg_mask.mask, phase=True)

call(["audacious", bg_phase.embed_audio().filename])
call(["audacious", fg_phase.embed_audio().filename])
fg_phase.embed_audio().filename


ft2d = nussl.separation.primitive.FT2D(mix)
ft2d_bg, ft2d_fg = ft2d()

call(["audacious", ft2d_fg.embed_audio().filename])
call(["audacious", ft2d_bg.embed_audio().filename])
masked = apply_mask(mix, ft2d_fg.mask)

testen = nussl.separation.spatial.Projet(mix, num_sources=2, device='cuda')
testen = nussl.separation.primitive.HPSS(mix)
test = testen(); print(len(test))
test0, *a = test
test0
len(test)

call(["audacious", test[1].embed_audio().filename])

lmix = mix.make_audio_signal_from_channel(0)
testen = nussl.separation.deep.DeepClustering(mix, model_path="slakhv0_9bDQSFk.pth", device='cuda', num_sources=2)

modellera = "slakhv0_9bDQSFk.pth"
modellera = "musdbv0_hnQwdQ4.pth"
modellera = "musdb+slakhv0_TG3EvX6.pth"
dc = []
for i in range(2):
    imix = mix.make_audio_signal_from_channel(i)
    testen = nussl.separation.deep.DeepClustering(imix, model_path=modellera, device='cuda', num_sources=2)
    test = testen(); print(len(test))
    dc.append(test)

rrs = []
for r in zip(*dc):
    newdat = np.concatenate([r[0].audio_data, r[1].audio_data],axis=0)
    result = rechan[0][0].make_copy_with_audio_data(newdat)
    rrs.append(result)

path = lyssna(rrs[1])
call(["scp", path, "thepgit:public_html/x_isol.mp3"])

remap = [[0,1,2,3], [3, 0, 2, 1]]
rs = []
for r in zip(*remap):
    newdat = np.concatenate([dc[0][r[0]].audio_data, dc[1][r[1]].audio_data],axis=0)
    result = rechan[0][0].make_copy_with_audio_data(newdat)
    rs.append(result)

lyssna(rs[3])

nussl.core.audio_signal
model_path = nussl.utils.download_trained_model('slakhv0_9bDQSFk.pth')
