from copy import deepcopy
from typing import List, Union

import torch
from transformers import PreTrainedModel


def model_to_devices(model: PreTrainedModel, *devices: Union[str, int]) -> List[PreTrainedModel]:
    devices = multi_devices(*devices)
    models = []
    model_device = parse_device(model.device)
    for device in devices:
        if device == model_device:
            models.append(model)
        else:
            models.append(deepcopy(model).to(device))
    return models


def multi_devices(*devices: Union[str, int]) -> List[str]:
    if not devices:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = [parse_device(device) for device in devices]
    return devices


def parse_device(device: Union[str, int, torch.device, None] = None) -> str:
    if isinstance(device, torch.device):
        return f"{device.type}:{device.index}"
    if device is None:
        if not torch.cuda.is_available():
            return "cpu"
        device = "cuda"
    if device == "cuda":
        return f"cuda:{torch.cuda.current_device()}"

    if isinstance(device, str):
        try:
            device = int(device)
        except Exception:
            pass
    if isinstance(device, int):
        return f"cuda:{device}"
    return device
