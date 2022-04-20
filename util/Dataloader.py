import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, label,len_crop=176,device='cuda:0'):
        self.data = data
        self.label = label
        self.len_crop = len_crop
        self.device = device

  
    def __getitem__(self, index):
        tmp = self.data[index].detach().numpy()
        if tmp.shape[0] < self.len_crop:
            pad_size = int(self.len_crop - tmp.shape[0])
            npad = [(0, 0)] * tmp.ndim
            npad[0] = (0, pad_size)
            tmp = np.pad(tmp, pad_width=npad, mode='constant', constant_values=0)
            melsp = torch.from_numpy(tmp)
            
        elif tmp.shape[0] == self.len_crop:
            melsp = torch.from_numpy(tmp)
        else:
            left = np.random.randint(0, tmp.shape[0] - self.len_crop)
            melsp = torch.from_numpy(tmp[left : left + self.len_crop, :])
        return melsp.to(self.device), self.label[index]

    def __len__(self):
        return len(self.data)


def generate_dataset(
    len_crop, num_uttrs, rootDir="../spmel", batch_size=64, use_shuffle=True
):
    dirName, subdirList, _ = next(os.walk(rootDir))
    print(f"Found directory: {dirName}")
    all_data, all_label = [], []
    for j, speaker in enumerate(sorted(subdirList)):
        print("Processing speaker: %s" % speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
        fileList = fileList[:num_uttrs]
        idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
        for i in range(num_uttrs):
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
            melsp = torch.from_numpy(tmp[np.newaxis, :, :])
            all_data.append(melsp.squeeze())
            all_label.append(torch.LongTensor([j]))
    
    train_dataset = MyDataset(
        all_data,
        torch.Tensor([item.cpu().detach().numpy() for item in all_label]).cuda(),
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=use_shuffle
    )
    return train_loader