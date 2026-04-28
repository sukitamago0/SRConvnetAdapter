import torch


def mask_adapter_cond(cond, keep_mask: torch.Tensor):
    if cond is None:
        return None
    if not torch.is_tensor(keep_mask):
        keep_mask = torch.tensor(keep_mask)

    def _find_device_dtype(x):
        if torch.is_tensor(x):
            return x.device, x.dtype
        if isinstance(x, dict):
            for v in x.values():
                found = _find_device_dtype(v)
                if found is not None:
                    return found
        if isinstance(x, (list, tuple)):
            for item in x:
                found = _find_device_dtype(item)
                if found is not None:
                    return found
        return None

    found = _find_device_dtype(cond)
    if found is None:
        return cond
    dev, _ = found
    keep_mask = keep_mask.to(device=dev, dtype=torch.float32)

    def _mask_tensor(x):
        m = keep_mask
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        return x * m.to(dtype=x.dtype)

    def _mask_obj(x):
        if torch.is_tensor(x):
            return _mask_tensor(x)
        if isinstance(x, dict):
            return {k: _mask_obj(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_mask_obj(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_mask_obj(v) for v in x)
        return x

    return _mask_obj(cond)
