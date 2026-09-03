import torch


def get_select_mask(tensor, skip_ratio=0, rand=False):
    # Use tensor operations for efficiency
    retain_mask = (tensor == -1).clone()
    unique_vals, counts = torch.unique(tensor, return_counts=True)

    for i, (val, count) in enumerate(zip(unique_vals, counts)):
        if val == -1:
            continue
        positions = (tensor == val).nonzero(as_tuple=True)[0]
        num_positions = len(positions)
        
        if num_positions == 1:
            retain_mask[positions] = True
        else:
            num_to_skip = int(round(num_positions * skip_ratio))
            num_to_retain = max(1, num_positions - num_to_skip)
            if rand:
                # rand means random select subset of selective tokens for layer-wise
                perm = torch.randperm(num_positions, device=tensor.device)
                positions_to_retain = positions[perm[:num_to_retain]]
            else:
                indices = torch.linspace(
                    0, num_positions - 1, steps=num_to_retain
                ).long()
                positions_to_retain = positions[indices]
                
            retain_mask[positions_to_retain] = True
    return retain_mask


def select_attention_mask(
    attention_mask,
    retain_mask,
    sequence_length=None,
    past_length=0,
    compact_key_axis=True,
):
    """Project an attention mask onto the tokens kept by sparse self-attention.

    UI-guided attention shortens the query and (during prefill) key sequences. A
    mask built for the original sequence therefore has to be indexed on both
    axes. ``past_length`` accounts for cached keys when a dynamic cache is
    compacted. Static caches keep their full key axis, so callers can disable
    ``compact_key_axis`` for that case.

    ``retain_mask`` accepts the boolean mask produced by :func:`get_select_mask`
    as well as an explicit one-dimensional index tensor. The latter is useful
    for callers that precompute selected token indices.
    """
    if attention_mask is None:
        return None

    if retain_mask.ndim != 1:
        raise ValueError("retain_mask must be one-dimensional")

    if retain_mask.dtype == torch.bool:
        source_length = retain_mask.numel()
        if sequence_length is not None and source_length != sequence_length:
            raise ValueError("retain_mask length must match sequence_length")
        indices = retain_mask.to(device=attention_mask.device).nonzero(as_tuple=True)[0]
    else:
        if sequence_length is None:
            raise ValueError("sequence_length is required for index-based retention")
        source_length = sequence_length
        indices = retain_mask.to(device=attention_mask.device, dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= source_length):
            raise IndexError("retain indices must fall within the source sequence")

    if indices.numel() == 0:
        raise ValueError("sparse attention must retain at least one token")
    if past_length < 0:
        raise ValueError("past_length must be non-negative")

    # Avoid an allocation on the common no-op path. This also keeps the exact
    # mask object used by dense attention when no token is removed.
    is_identity = indices.numel() == source_length and torch.equal(
        indices, torch.arange(source_length, device=indices.device, dtype=indices.dtype)
    )
    if is_identity:
        return attention_mask

    projected = attention_mask
    if compact_key_axis:
        if past_length:
            prefix = torch.arange(past_length, device=indices.device)
            key_indices = torch.cat((prefix, past_length + indices))
        else:
            key_indices = indices
    else:
        key_indices = indices

    def project_key_axis(mask):
        key_length = mask.shape[-1]
        if compact_key_axis:
            # A larger axis can contain a cache tail or an extra target slot;
            # select only positions that correspond to actual key states.
            if key_indices.numel() and int(key_indices.max()) < key_length:
                return mask.index_select(-1, key_indices)
        elif key_length == source_length:
            return mask.index_select(-1, key_indices)
        return mask

    if projected.ndim == 1:
        return project_key_axis(projected)

    # A 2-D mask is the format consumed by FlashAttention. Its final axis is
    # the key sequence; the query length is supplied separately by the caller.
    if projected.ndim == 2:
        return project_key_axis(projected)

    # Eager and SDPA paths use [batch, heads, query, key] (or the equivalent
    # three-dimensional form). Query rows always describe the current input.
    # Key columns are projected according to the cache mode above.
    if projected.shape[-2] == source_length:
        projected = projected.index_select(-2, indices)
    return project_key_axis(projected)
