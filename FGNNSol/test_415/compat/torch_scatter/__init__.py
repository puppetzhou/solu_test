import torch


def _infer_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    if dim != 0:
        raise NotImplementedError("compat torch_scatter only supports dim=0")
    dim_size = _infer_dim_size(index, dim_size)
    out_shape = (dim_size,) + tuple(src.shape[1:])
    if out is None:
        out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    expand_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, expand_index, src)
    return out


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    if dim != 0:
        raise NotImplementedError("compat torch_scatter only supports dim=0")
    summed = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    counts = torch.bincount(index, minlength=summed.size(0)).clamp_min(1)
    view_shape = (counts.size(0),) + (1,) * (src.dim() - 1)
    return summed / counts.view(view_shape).to(src.dtype)
