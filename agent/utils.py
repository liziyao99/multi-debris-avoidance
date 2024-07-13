import torch
import typing

def _split_by_values(tensor:torch.Tensor, sub_tensors:typing.List[torch.Tensor]=[]):
    '''
        args:
            `tensor`: 1D tensor.
            `sub_tensors`: list of 1D tensors.
    '''
    last_idx = 0
    current_value = None
    devided = []
    
    for i in range(tensor.shape[0]):
        if tensor[i]!=current_value:
            if current_value is not None:
                temp = [tensor[last_idx:i]]
                for sub_tensor in sub_tensors:
                    temp.append(sub_tensor[last_idx:i])
                devided.append(temp)
                last_idx = i
            current_value = tensor[i]

    temp = [tensor[last_idx:]]
    for sub_tensor in sub_tensors:
        temp.append(sub_tensor[last_idx:])
    devided.append(temp)
    last_idx = i
    
    return devided

def var_len_seq_sort(seqs:typing.List[torch.Tensor]):
    """
        Sort a batch of sequences by length in descending order.
        args:
            `seqs`: each element is a 2D tensor with shape (seq_len, feature_dim).
        returns:
            `stacked`: a list of 3D tensors with shape (_size, seq_len, feature_dim).
            `indices`: a list of 1D tensors with shape (_size,).
    """
    len, idx = torch.sort(torch.tensor([x.shape[0] for x in seqs]))
    devided = _split_by_values(len, [idx])
    stacked = []
    indices = []
    for d in devided:
        temp = torch.stack([seqs[i] for i in d[1]], dim=0)
        stacked.append(temp)
        indices.append(d[1])
    return stacked, indices