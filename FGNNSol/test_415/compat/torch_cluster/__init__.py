import torch


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, num_workers=1):
    if batch is not None:
        raise NotImplementedError("compat torch_cluster.radius_graph does not support batch")
    if x.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    dist = torch.cdist(x, x)
    mask = dist <= r
    if not loop:
        mask &= ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)

    if max_num_neighbors is not None and max_num_neighbors > 0:
        edges = []
        for dst in range(x.size(0)):
            src_idx = torch.nonzero(mask[:, dst], as_tuple=False).view(-1)
            if src_idx.numel() > max_num_neighbors:
                local_dist = dist[src_idx, dst]
                keep = torch.argsort(local_dist)[:max_num_neighbors]
                src_idx = src_idx[keep]
            if src_idx.numel():
                dst_idx = torch.full_like(src_idx, dst)
                edges.append(torch.stack((src_idx, dst_idx), dim=0))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)
        return torch.cat(edges, dim=1)

    return torch.nonzero(mask, as_tuple=False).t().contiguous()
