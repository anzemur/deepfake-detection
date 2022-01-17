import torch

def get_device(gpu_idx=None):
  return torch.device((f"cuda:{str(gpu_idx)}" if gpu_idx else "cuda") if torch.cuda.is_available() else "cpu")