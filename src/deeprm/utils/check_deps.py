def check_torch_available():
    """
    Checks if PyTorch is available.
    Raises:
        SystemExit: If PyTorch is not installed, provides instructions for installation.
    Returns:
        None
    """
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "Torch is not installed. "
            "For CPU: `pip install 'deeprm[torch]'`."
            "For GPU: install torch with the appropriate CUDA index URL, then re-run."
        ) from e
    return None


def check_torchmetrics_available():
    """
    Checks if torchmetrics is available.
    Raises:
        SystemExit: If torchmetrics is not installed, provides instructions for installation.
    Returns:
        None
    """
    try:
        import torchmetrics
    except ImportError as e:
        raise SystemExit(
            "Torchmetrics is not installed.\n" "Please install it with `pip install 'torchmetrics'`.\n"
        ) from e
    return None
