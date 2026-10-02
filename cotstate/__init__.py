import os, sys, ctypes

_cu13_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv/lib/python3.10/site-packages/nvidia/cu13/lib")
if os.path.isdir(_cu13_dir):
    _curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _cu13_dir not in _curr_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{_cu13_dir}:{_curr_ld}" if _curr_ld else _cu13_dir
    for _lib in ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libnvJitLink.so.13", "libnvrtc.so.13"]:
        _p = os.path.join(_cu13_dir, _lib)
        if os.path.exists(_p):
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
