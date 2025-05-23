import torch
import torch.nn.functional as F


def compute_ll_clm(logits: torch.Tensor, input_ids: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate log-likelihood (LL) for causal LM.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    if reduction == 'mean':
        ll = log_likelihood.mean(dim=-1)
    elif reduction == 'sum':
        ll = log_likelihood.sum(dim=-1)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    return ll


def compute_llr_mlm(logits: torch.Tensor, input_ids: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate log-likelihood MLM.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    if reduction == 'mean':
        llr = log_likelihood.mean(dim=-1)
    elif reduction == 'sum':
        llr = log_likelihood.sum(dim=-1)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    return llr
