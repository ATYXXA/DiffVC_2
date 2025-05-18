import os
from tqdm import tqdm
import json
import numpy as np
import torch
import params
from model.vc import DiffVC_Plus
from utils import save_audio
from model.utils import repeat_expand_2d

import librosa
from librosa.filters import mel as librosa_mel_fn
sampling_rate = 22050
mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

out_dir = 'outputs'
vc_path = 'logs_dec/vc_100.pt'

from model.ContentVec768L12 import ContentVec768L12
cnt_encoder = None

from model.eg import VAE
eg = None

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
hifigan_universal = None

def get_mel(wav_path):
    wav, _ = librosa.load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_unit(wav_path):
    wav, _ = librosa.load(wav_path, sr=16000)
    wav16k = torch.from_numpy(wav).to(cnt_encoder.dev)
    unit = cnt_encoder.encoder(wav16k).squeeze()
    # unit = unit.detach().cpu().numpy()
    unit = unit.detach()
    return unit

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') 
spk_encoder.load_model(enc_model_fpath, device="cuda")
def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i+2*w+1])
        y[i] = min(x[i+w+1], med)
    return y

def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


def process_one(src_path, tgt_path):
    src_name = src_path.split('/')[-1].split('.')[0]
    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    mel_source = mel_source.cuda()
    
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    mel_source_lengths = mel_source_lengths.cuda()

    unit_source = get_unit(src_path)
    unit_source = repeat_expand_2d(unit_source,target_len=mel_source.shape[-1]).unsqueeze(0)
    unit_source = unit_source.cuda()

    if eg:
        noise = torch.randn(1, 64).cuda()
        generated_emb = eg.Decoder(noise)
        tgt_emb=generated_emb.detach()
    else:
        tgt_emb = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)

    mel_avg, mel_rec = model(mel_source, mel_source_lengths, mel_source, mel_source_lengths, tgt_emb, unit_source, 
                             n_timesteps=50) # 30
    mel_synth_np = mel_rec.detach().cpu().squeeze().numpy()
    mel_source_np = mel_rec.detach().cpu().squeeze().numpy()  
    mel_output = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)   
    audio = hifigan_universal.forward(mel_output.cuda())

    if eg:
        tgt = "eg"
    else: 
        tgt = os.path.basename(tgt_path).split('_')[0]
    os.makedirs(f'{out_dir}/{tgt}', exist_ok=True)
    out_name = f'{src_name}_to_{tgt}'
    save_audio(f'{out_dir}/{tgt}/{out_name}.wav', sampling_rate, audio, normalize=True)                        


def search_wavs(rootdir):
    paths=[]
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".wav"):
                paths.append(os.path.join(root,file))
    
    return paths

def process_dir(dir):
    print("Processing...")
    for wav in tqdm(search_wavs(dir)):
        process_one(wav)

def process_list(filepath):
    with open(filepath,'r') as ff:
        lines=ff.readlines()
    for line in tqdm(lines):
        src, tgt = line.strip().split('|')
        process_one(src, tgt)


if __name__ == "__main__":

    os.makedirs(out_dir, exist_ok=True)

    print('Initializing and loading models...')
    model = DiffVC_Plus(params.n_mels, params.spk_dim, params.dec_dim,
                   params.beta_min, params.beta_max)

    print(f'Loading ckpt from {vc_path}.')
    model = model.cuda()
    model.load_state_dict(torch.load(vc_path))
    model.eval()

    print('Decoder - Number of parameters = %.2fm' % (model.decoder.nparams/1e6))
    torch.backends.cudnn.benchmark = True

   
    cnt_encoder = ContentVec768L12(device="cuda")

    eg = VAE().cuda()
    eg.load_state_dict(torch.load('psg_mylossf_50.pt'))
    eg.eval()

    hfg_path = 'checkpts/vocoder/' 
    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))
    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
    hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()    

    with torch.no_grad():
        process_list('path/to/vcpair.txt') # 1 line in vcpair.txt: path/to/src.wav|path/to/tgt.wav  
                    