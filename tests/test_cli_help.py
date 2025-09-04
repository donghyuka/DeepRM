import subprocess


def test_cli_top_level_help():
    # Ensure the console script is installed and responds to --help
    result = subprocess.run(["deeprm", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DeepRM" in result.stdout or "DeepRM" in result.stderr
    assert "usage" in result.stdout.lower()


def test_cli_inference_help():
    # Test the inference CLI help
    result = subprocess.run(["deeprm", "call", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DeepRM Call (inference) Module" in result.stdout or result.stderr
    assert "usage" in result.stdout.lower()


def test_cli_train_help():
    # Test the training CLI help
    result = subprocess.run(["deeprm", "train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DeepRM Training Module" in result.stdout or result.stderr
    assert "usage" in result.stdout.lower()


def test_cli_qc_help():
    # Test the QC CLI help
    result = subprocess.run(["deeprm", "qc", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DeepRM QC Module" in result.stdout or result.stderr
    assert "usage" in result.stdout.lower()
