import librosa
import matplotlib.pyplot as plt
import numpy as np


audio_path = "D:\\farooq\\file_example_WAV_1MG.wav"  # put any .wav file here

y, sr = librosa.load(audio_path, sr=16000)

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)


plt.imshow(mel_db, aspect='auto', origin='lower')
plt.title("Mel Spectrogram")
plt.colorbar()
plt.show()
