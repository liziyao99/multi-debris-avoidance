import numpy as np
import torch

class npzLoader:
    def __init__(self, file:str) -> None:
        self.file = file

    @classmethod
    def make(cls, file, feature, label):
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        np.savez(file, feature=feature, label=label)
        return cls(file)

    def torch(self, device=None):
        data = np.load(self.file)
        feature = torch.from_numpy(data['feature']).float().to(device)
        label = torch.from_numpy(data['label']).float().to(device)
        return feature, label
    
    def numpy(self):
        data = np.load(self.file)
        feature = data['feature']
        label = data['label']
        return feature, label