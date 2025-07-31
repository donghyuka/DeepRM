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
