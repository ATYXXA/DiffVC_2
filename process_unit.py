import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import librosa
import torch

from model.ContentVec768L12 import ContentVec768L12
cnt_encoder = ContentVec768L12(device="cuda")

def get_unit(wav_path):
    wav, _ = librosa.load(wav_path, sr=16000)
    wav16k = torch.from_numpy(wav).to(cnt_encoder.dev)
    unit = cnt_encoder.encoder(wav16k).squeeze()
    unit = unit.detach().cpu().numpy()
    return unit

def process_one(wav_path):
    unit = get_unit(wav_path)
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_name = os.path.basename(os.path.dirname(wav_path))
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'units', f'{speaker_name}', f"{file_name}_unit.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, unit)


def process_all(root_folder, num_processes):
    wav_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(root, file))

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_one, wav_paths), total=len(wav_paths)))
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    data_folder = "dataset/vctk"  
    num_processes = 4  
    multiprocessing.set_start_method('spawn', force=True)
    process_all(data_folder, num_processes)

