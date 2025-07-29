import subprocess, sys

def test_cli_top_level_help():
    # Ensure the console script is installed and responds to --help
    proc = subprocess.run([sys.executable, "-m", "deeprm", "--help"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0
    assert "DeepRM" in proc.stdout or "DeepRM" in proc.stderr
