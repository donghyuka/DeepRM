import subprocess
import sys


def test_cli_top_level_help():
    # Ensure the console script is installed and responds to --help
    result = subprocess.run([sys.executable, "-m", "deeprm", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DeepRM" in result.stdout or "DeepRM" in result.stderr
    assert "usage" in result.stdout.lower()
