import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing

mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

def get_mel(wav_path):
    wav, _ = librosa.load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def process_one(wav_path):
    log_mel_spectrogram = get_mel(wav_path)

    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_name = os.path.basename(os.path.dirname(wav_path))
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'mels', f'{speaker_name}', f"{file_name}_mel.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.save(save_path, log_mel_spectrogram)

def process_all(root_folder, num_processes):
    wav_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(root, file))

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_one, wav_paths), total=len(wav_paths)))

if __name__ == "__main__":
    data_folder = "dataset/vctk" 
    num_processes = 4  
    multiprocessing.set_start_method('spawn', force=True)
    process_all(data_folder, num_processes)

