from torch.nn.utils.rnn import pad_sequence
import torch
class DataCollatePad:
    def __init__(self, pad_ids = [0], without_pad_idxs=[2]):
        self.pad_ids = pad_ids
        self.without_pad_idxs = without_pad_idxs
    
    def __call__(self, features):
        rtn_features = []
        for i in range(len(features[0])):
            if i in self.without_pad_idxs:
                rtn_features.append(torch.stack([f[i] for f in features]))
            else:
                rtn_features.append(pad_sequence([f[i] for f in features], batch_first=True, padding_value=self.pad_ids[i]))

        return tuple(rtn_features)
