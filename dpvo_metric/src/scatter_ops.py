from __future__ import annotations
"""
Native PyTorch implementations of scatter operations.

Replaces torch_scatter dependency with equivalent PyTorch operations.
"""

import torch


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None
) -> torch.Tensor:
    """
    Sum values from src into output at positions specified by index.
    
    Equivalent to torch_scatter.scatter_sum.
    
    Args:
        src: Source tensor
        index: Index tensor (same shape as src along dim, or broadcastable)
        dim: Dimension along which to scatter
        dim_size: Size of output along dim (default: max(index) + 1)
    
    Returns:
        Tensor with summed values at indexed positions
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    # Build output shape
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # Expand index to match src shape for scatter_add
    index_expanded = _expand_index(index, src, dim)
    
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add(dim, index_expanded, src)


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand index tensor to match source tensor shape."""
    if index.dim() == src.dim():
        return index.expand_as(src)
    
    # index has fewer dimensions - need to reshape and expand
    # First, add dimensions to match src
    view_shape = [1] * src.dim()
    view_shape[dim] = index.numel()
    index = index.view(view_shape)
    
    # Now expand to match src
    expand_shape = list(src.shape)
    expand_shape[dim] = index.shape[dim]
    return index.expand(expand_shape)


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find maximum values from src at positions specified by index.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to scatter
        dim_size: Size of output along dim
    
    Returns:
        Tuple of (max values, argmax indices)
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    index_expanded = _expand_index(index, src, dim)
    
    # Use scatter_reduce with 'amax' for max operation
    out = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim, index_expanded, src, reduce='amax', include_self=False)
    
    # Replace -inf with 0 for positions with no values
    out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
    
    # Argmax not needed for our use case, return dummy
    return out, torch.zeros_like(out, dtype=torch.long)


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None
) -> torch.Tensor:
    """
    Compute softmax over groups defined by index.
    
    Equivalent to torch_scatter.scatter_softmax.
    
    Args:
        src: Source tensor
        index: Index tensor defining groups
        dim: Dimension along which to compute softmax
        dim_size: Size of the grouping dimension
    
    Returns:
        Softmax values within each group
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    index_expanded = _expand_index(index, src, dim)
    
    # Get max per group for numerical stability
    max_vals, _ = scatter_max(src, index, dim, dim_size)
    src_centered = src - max_vals.gather(dim, index_expanded)
    
    # Compute exp
    src_exp = src_centered.exp()
    
    # Sum exp per group
    sum_exp = scatter_sum(src_exp, index, dim, dim_size)
    
    # Normalize
    return src_exp / sum_exp.gather(dim, index_expanded).clamp(min=1e-12)
