import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing
import sys
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
spk_encoder.load_model(enc_model_fpath, device="cuda")

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

def process_one(wav_path):
    embded = get_embed(wav_path)

    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_name = os.path.basename(os.path.dirname(wav_path))
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'embeds', f'{speaker_name}', f"{file_name}_embed.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.save(save_path, embded)


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
