import torch
import numpy as np


def list_to_padded(list_of_tensors, max_len, axis, pad_value=0):

    # Pad a list of tensors to the same length along a given axis.
    full_shape = list_of_tensors[0].size()
    full_shape = list(full_shape)
    full_shape[axis] = max_len
    full_shape[0] = len(list_of_tensors)
    padded_tensor = torch.full(full_shape, pad_value, dtype=list_of_tensors[0].dtype).to(list_of_tensors[0].device)
    for i, tensor in enumerate(list_of_tensors):
        if axis == 1:
            padded_tensor[i, :tensor.size(axis)] = tensor
        elif axis == 2:
            padded_tensor[i, :, :tensor.size(axis)] = tensor
        elif axis == 3:
            padded_tensor[i, :, :, :tensor.size(axis)] = tensor
        else:
            raise ValueError("Only 3D tensors are supported.")            
    return padded_tensor


def cat(list_of_tensor_or_array, dim):
    if isinstance(list_of_tensor_or_array[0], torch.Tensor):
        return torch.cat(list_of_tensor_or_array, dim)
    elif isinstance(list_of_tensor_or_array[0], np.ndarray):
        return np.concatenate(list_of_tensor_or_array, dim)
    elif isinstance(list_of_tensor_or_array[0], (list, tuple)):
        res = []
        for v in list_of_tensor_or_array:
            res += v
        return res
    
def concat_dict_list(list_of_dict):
    output = dict()
    keys = list_of_dict[0].keys()
    for k in keys:
        if isinstance(list_of_dict[0][k], torch.Tensor):
            selected_tensors = [curr_dict[k] for curr_dict in list_of_dict]
            output[k] = cat(selected_tensors, 0)
        elif isinstance(list_of_dict[0][k], np.ndarray):
            selected_tensors = [curr_dict[k] for curr_dict in list_of_dict]
            output[k] = cat(selected_tensors, 0)
        elif isinstance(list_of_dict[0][k], (list, tuple)):
            selected_tensors = [curr_dict[k] for curr_dict in list_of_dict]
            output[k] = cat(selected_tensors, 0)
        elif isinstance(list_of_dict[0][k], dict):
            output[k] = concat_dict_list(
                [curr_dict[k] for curr_dict in list_of_dict])
    return output

def slice_dict(input_dict, index, unwanted_keys=(), wanted_keys=()):
    out_dict = dict()
    if wanted_keys:
        unwanted_keys = set(input_dict.keys()) - set(wanted_keys)
    elif unwanted_keys:
        wanted_keys = set(input_dict.keys()) - set(unwanted_keys)
    else:
        wanted_keys = input_dict.keys()
    if isinstance(index, (np.ndarray, torch.Tensor)):
        index = index.tolist()
    elif isinstance(index, int):
        index = [index]
    for k, v in input_dict.items():
        if hasattr(v, '__getitem__') and not isinstance(
                v, str) and not (isinstance(
                    v, dict)):
            if k in wanted_keys:
                if isinstance(v, (torch.Tensor, np.ndarray)) and not v.shape:
                    out_dict[k] = v

                elif len(v) == 1:
                    out_dict[k] = v
                elif isinstance(v, (list, tuple)):
                    out_dict[k] = [v[i] for i in index]
                else:
                    out_dict[k] = v[index]
        elif isinstance(v, (list, tuple)): # get the index element of the list
            out_dict[k] = [v[i] for i in index]
        elif isinstance(v, dict):
            out_dict[k] = slice_dict(v, index)
        else:
            out_dict[k] = v
    return out_dict

def dict2tensor(input_dict, device=None, use_float=True):
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            try:
                if use_float:
                    input_dict[k] = torch.tensor(v).float()
                else:
                    input_dict[k] = torch.tensor(v)
                if device is not None:
                    input_dict[k] = input_dict[k].to(device)
            except:
                pass
        elif isinstance(v, torch.Tensor):
            if use_float:
                input_dict[k] = v.float()
            if device is not None:
                input_dict[k] = v.to(device)

        elif isinstance(v, dict):
            input_dict[k] = dict2tensor(v, device, use_float)

    return input_dict