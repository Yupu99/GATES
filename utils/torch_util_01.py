import torch
import numpy as np


# ---------- Basic: read only trainable parameters ----------
def get_param_list(model):
    """
    Return a list of all trainable parameters (requires_grad=True) as numpy arrays.
    Each array is moved to CPU and copied to avoid referencing the original tensor.
    """
    param_lst = []
    for p in model.parameters():
        if not p.requires_grad:
            # Optional: print for debugging
            # print('skip frozen param:', p.shape)
            continue
        param_lst.append(p.detach().cpu().numpy().copy())
    return param_lst


# ---------- Write back trainable parameters ----------
def set_param_list(model, param_lst):
    """
    Write a list of numpy arrays back into the model parameters
    (only for requires_grad=True parameters, order must match get_param_list).
    Keeps device and dtype consistent.
    """
    lst_idx = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            new_np = param_lst[lst_idx]
            # Check shape consistency
            assert tuple(new_np.shape) == tuple(p.shape), \
                f"shape mismatch: {new_np.shape} vs {tuple(p.shape)}"
            new_t = torch.from_numpy(new_np).to(p.device).type_as(p)
            p.copy_(new_t)
            lst_idx += 1
    # Safety check: all given arrays must be used
    assert lst_idx == len(param_lst), f"set count {lst_idx} != given {len(param_lst)}"


# ---------- Flatten / Unflatten trainable parameters ----------
def get_flatten_params(model):
    """
    Flatten all trainable parameters (requires_grad=True) into a 1D numpy vector.
    Returns both the flat vector and slice indices to reconstruct later.
    """
    param_list = get_param_list(model)
    flats = [np.ravel(p) for p in param_list]
    lengths = []
    s = 0
    for fp in flats:
        e = s + fp.size
        lengths.append((s, e))
        s = e
    flat = np.concatenate(flats, axis=0) if len(flats) > 0 else np.array([], dtype=np.float32)
    return {"params": flat, "lengths": lengths}


def set_flatten_params(flat_params, lengths, model):
    """
    Restore parameters from a flattened vector using slice indices.
    Only affects trainable parameters (requires_grad=True).
    """
    # Validation
    if len(lengths) == 0:
        return
    assert lengths[-1][1] == flat_params.size, \
        f"flat size {flat_params.size} != expected {lengths[-1][1]}"

    # Collect current trainable parameters as shape templates
    current = [p for p in model.parameters() if p.requires_grad]
    assert len(current) == len(lengths), \
        f"trainable count changed: now {len(current)} vs meta {len(lengths)}"

    # Slice the flat vector and reshape each block
    param_blocks = []
    for (s, e), p in zip(lengths, current):
        block = flat_params[s:e].reshape(p.shape)
        param_blocks.append(block.copy())

    # Write back
    set_param_list(model, param_blocks)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


@torch.no_grad()
def xavier_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.fill_(0.0)


