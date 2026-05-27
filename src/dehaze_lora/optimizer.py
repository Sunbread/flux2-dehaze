import torch


def create_optimizer(model, lr: float = 1e-3, weight_decay: float = 0.01, momentum: float = 0.95):
    """Muon optimizer for a single model's LoRA parameters.

    Collects all trainable (requires_grad) parameters from the model
    and wraps them in a Muon optimizer. LoRA A/B matrices are 2D so
    they satisfy Muon's 2D constraint.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]

    if not trainable:
        raise ValueError("No trainable parameters found in model")

    non_2d = [p for p in trainable if p.ndim != 2]
    if non_2d:
        raise ValueError(
            f"Muon only supports 2D parameters, got {len(non_2d)} non-2D: "
            + ", ".join(f"{tuple(p.shape)}" for p in non_2d)
        )

    return torch.optim.Muon(
        trainable,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        adjust_lr_fn="match_rms_adamw",
    )
