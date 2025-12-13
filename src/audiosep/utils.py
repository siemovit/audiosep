import torch

def center_crop(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """x: (..., T) -> (..., out_len), taking the center part."""
    T = x.shape[-1]
    if out_len > T:
        raise ValueError(f"out_len ({out_len}) > T ({T})")
    start = (T - out_len) // 2
    return x[..., start:start + out_len]