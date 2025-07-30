from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from deeprm.utils.logging import get_logger

log = get_logger(__name__)


# Optional color (works even on Windows via colorama if installed)
def _supports_color(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


@dataclass
class Issue:
    severity: str  # "ERROR" | "WARN"
    message: str
    hint: str | None = None
    code: str | None = None  # machine-usable code, e.g., "TORCH_MISSING"


def _run(cmd: List[str], timeout: int = 6) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 127, "", str(e)


def _parse_nvidia_smi_banner(txt: str) -> Tuple[str | None, str | None]:
    # Looks for lines like: "CUDA Version: 12.4" and "Driver Version: 535.129.03"
    drv = None
    cuda = None
    # First line often contains both:
    m_drv = re.search(r"Driver Version:\s*([0-9.]+)", txt)
    if m_drv:
        drv = m_drv.group(1)
    m_cuda = re.search(r"CUDA Version:\s*([0-9.]+)", txt)
    if m_cuda:
        cuda = m_cuda.group(1)
    return drv, cuda


def _nvidia_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"present": False}
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return info
    info["present"] = True
    rc, out, _ = _run([nvsmi])
    if rc == 0:
        drv, cuda = _parse_nvidia_smi_banner(out)
        if drv:
            info["driver_version"] = drv
        if cuda:
            info["driver_supports_cuda"] = cuda
    # Also query GPUs count (optional)
    rc2, out2, _ = _run([nvsmi, "--query-gpu=name", "--format=csv,noheader"])
    if rc2 == 0:
        names = [ln.strip() for ln in out2.splitlines() if ln.strip()]
        info["gpus"] = names
        info["gpu_count"] = len(names)
    return info


def _rocm_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"present": False}
    rocminfo = shutil.which("rocminfo") or shutil.which("rocminfo.py")
    rocmsmi = shutil.which("rocm-smi") or shutil.which("rocm-smi.py")
    if rocminfo or rocmsmi:
        info["present"] = True
        if rocmsmi:
            rc, out, _ = _run([rocmsmi, "-v"])
            if rc == 0:
                m = re.search(r"ROCm\s+Version\s*:\s*([0-9.]+)", out)
                if m:
                    info["rocm_version"] = m.group(1)
    return info


def _torch_info() -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["present"] = True
        info["version"] = getattr(torch, "__version__", None)
        info["built_cuda"] = getattr(torch.version, "cuda", None)
        info["built_rocm"] = getattr(torch.version, "hip", None) or getattr(torch.version, "rocm", None)
        # Avoid crashing when CUDA libs missing
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
        except Exception:
            info["cuda_available"] = False
            info["cuda_device_count"] = 0
    except Exception as e:
        info["present"] = False
        issues.append(
            Issue(
                severity="ERROR",
                message=f"PyTorch is not importable ({e}).",
                hint=(
                    "Install PyTorch first for your platform from https://pytorch.org/get-started/ "
                    "then install DeepRM extras: `pip install 'deeprm[train,inference]'`. "
                    "CPU-only: `pip install 'deeprm[torch,train,inference]'`."
                ),
                code="TORCH_MISSING",
            )
        )
        return info, issues

    # Built for CUDA but runtime not available
    if info.get("built_cuda") and not info.get("cuda_available"):
        issues.append(
            Issue(
                severity="ERROR",
                message=(
                    "You have a CUDA-built torch (compiled with CUDA {}) "
                    "but that version of CUDA is not available at runtime.".format(info["built_cuda"])
                ),
                hint=(
                    "This usually means the NVIDIA driver/CUDA runtime is missing or incompatible. "
                    "Check `nvidia-smi` output and reinstall torch for the correct CUDA version "
                    "(or install CPU torch if you don't need GPU)."
                ),
                code="CUDA_RUNTIME_MISSING",
            )
        )

    # CPU build on a GPU machine (notice-level -> WARN)
    nvi = _nvidia_info()
    if not info.get("built_cuda") and nvi.get("present"):
        issues.append(
            Issue(
                severity="WARN",
                message=("CPU-only torch detected while an NVIDIA driver is present."),
                hint=(
                    "If you intend to use the GPU, install a CUDA build of torch matching your driver, e.g.: "
                    "`pip install torch --index-url https://download.pytorch.org/whl/cu121`"
                ),
                code="CPU_TORCH_ON_GPU_MACHINE",
            )
        )

    # ROCm build but no rocm tools found
    rci = _rocm_info()
    if info.get("built_rocm") and not rci.get("present"):
        issues.append(
            Issue(
                severity="WARN",
                message=("ROCm-built torch detected but ROCm tools were not found on PATH."),
                hint=("Ensure ROCm runtime is installed and `rocminfo`/`rocm-smi` are available."),
                code="ROCM_RUNTIME_MISSING",
            )
        )

    return info, issues


def _torchmetrics_info(require_train: bool) -> Tuple[Dict[str, Any], List[Issue]]:
    info: Dict[str, Any] = {}
    issues: List[Issue] = []
    try:
        import torchmetrics  # type: ignore

        info["present"] = True
        info["version"] = getattr(torchmetrics, "__version__", None)
    except Exception as e:
        info["present"] = False
        sev = "ERROR" if require_train else "WARN"
        issues.append(
            Issue(
                severity=sev,
                message=f"torchmetrics is not importable ({e}).",
                hint=(
                    "Install with: `pip install 'deeprm[train]'` "
                    "after installing torch (GPU/ROCm users: install torch from the official index URL first)."
                ),
                code="TORCHMETRICS_MISSING",
            )
        )
    return info, issues


def collect(require_train: bool = False) -> Tuple[Dict[str, Any], List[Issue]]:
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    nvi = _nvidia_info()
    if nvi.get("present"):
        env["nvidia"] = nvi
    rci = _rocm_info()
    if rci.get("present"):
        env["rocm"] = rci

    torch_info, torch_issues = _torch_info()
    tmetrics_info, tmetrics_issues = _torchmetrics_info(require_train=require_train)

    env["torch"] = torch_info
    env["torchmetrics"] = tmetrics_info

    issues = torch_issues + tmetrics_issues
    return env, issues


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "deeprm doctor", description="Environment checks for DeepRM (torch/metrics/GPU)."
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Show all environment details.")
    p.add_argument(
        "--require-train", action="store_true", default=False, help="Check for training dependencies (torchmetrics)."
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = parser().parse_args(argv)
    env, issues = collect(require_train=args.require_train)
    warnings = [i for i in issues if i.severity == "WARN"]
    errors = [i for i in issues if i.severity == "ERROR"]

    if errors:
        log.error(f"Environment checks failed with {len(errors)} error{'s' if len(errors) > 1 else ''}:")
        for i, issue in enumerate(errors, start=1):
            log.error(f"{i}. {issue.code}")
            log.error(f"\t- {issue.message}")
            if issue.hint:
                log.error(f"\t- Hint: {issue.hint}")
        if warnings:
            log.warning(f"Additionally, {len(warnings)} warning{'s' if len(warnings) > 1 else ''} were found:")
            for i, issue in enumerate(warnings, start=1):
                log.warning(f"{i}. {issue.code}")
                log.warning(f"\t{issue.message}")
                if issue.hint:
                    log.warning(f"\tHint: {issue.hint}")

        log.warning("Re-run with `--verbose` flag to see full environment details.")

    elif warnings:
        log.warning(f"Environment checks completed with {len(warnings)} warning{'s' if len(warnings) > 1 else ''}:")
        for i, issue in enumerate(warnings, start=1):
            log.warning(f"{i}. {issue.code}")
            log.warning(f"\t{issue.message}")
            if issue.hint:
                log.warning(f"\tHint: {issue.hint}")

        log.warning("Re-run with `--verbose` flag to see full environment details.")

    else:
        log.info("Environment checks passed successfully.")

    if args.verbose:
        print(json.dumps(env, indent=2))

    return None


def entry(argv: List[str] | None = None) -> int:
    """Entry point for the CLI."""
    return main(argv) if argv is not None else main(sys.argv[1:])
