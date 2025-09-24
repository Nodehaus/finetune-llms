import torch


def get_gpu_info():
    """Get GPU information."""
    gpu_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_model": None,
        "vram_total_gb": None,
        "cuda_version": None,
    }

    if torch.cuda.is_available():
        try:
            gpu_info.update(
                {
                    "gpu_model": torch.cuda.get_device_name(0),
                    "vram_total_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                    ),
                    "cuda_version": torch.version.cuda,
                }
            )
        except Exception:
            pass

    return gpu_info
