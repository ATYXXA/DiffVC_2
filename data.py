import os
import random
import numpy as np
import torch
from params import seed as random_seed
from params import n_mels, train_frames
from model.utils import repeat_expand_2d


def get_vctk_unseen_speakers():
    unseen_speakers = ['p252', 'p261', 'p241', 'p238', 'p243',
                       'p294', 'p334', 'p343', 'p360', 'p362']
    return unseen_speakers


def get_vctk_unseen_sentences():
    unseen_sentences = ['001', '002', '003', '004', '005']
    return unseen_sentences


class VCTKDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unit_dir = os.path.join(data_dir, 'units')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mel(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        unit = self.get_unit(audio_id, spk, mels.shape[-1])
        return (mels, embed, unit)

    def get_mel(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()
        return mel

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed
    
    def get_unit(self, audio_id, spk, mel_len):
        unit_path = os.path.join(self.unit_dir, spk, audio_id + '_unit.npy')
        unit = np.load(unit_path)
        unit = torch.from_numpy(unit).float()
        unit = repeat_expand_2d(unit, target_len=mel_len, mode='nearest')
        return unit        


    def __getitem__(self, index):
        mels, embed, unit = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed, 'unit': unit}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        tripairs = []
        for i in range(len(self.valid_info)):
            mel, embed, unit= self.get_vc_data(self.valid_info[i])
            tripairs.append((mel, embed, unit))
        return tripairs
    
    def get_unseen_dataset(self):
        unseen_info = []
        spk_dict = {}
        utt_dict = {}
        for unseen_spk in self.unseen_speakers:
            spk_dict[unseen_spk] = []
        for unseen_spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, unseen_spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] in self.unseen_sentences]
            unseen_info += [(i[:-8], unseen_spk) for i in mel_ids]
            spk_dict[unseen_spk] += [i[:-8] for i in mel_ids]
        print("Total number of unseen wavs is %d." % len(unseen_info))
        print("Total number of unseen speakers is %d." % len(self.unseen_speakers))

        for i in range(len(unseen_info)):
            mels, embed, unit = self.get_vc_data(unseen_info[i])
            utt_dict[unseen_info[i][0]] = (mels, embed, unit)

        return spk_dict, utt_dict

class VCDecBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels1_unit = torch.zeros((B, 768, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            unit = item['unit']
            assert mel.shape[-1] == unit.shape[-1], "unit has not been expanded to match mel"
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels1_unit[i, :, :mel_length] = unit[:, starts1[i]:starts1[i] + mel_length]

            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'mel1': mels1, 'mel_lengths': mel_lengths, 'c': embed, 'mel1_unit': mels1_unit}
