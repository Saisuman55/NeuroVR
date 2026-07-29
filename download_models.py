"""
Download model weights from Hugging Face Hub at container startup.

Set the environment variable HF_MODEL_REPO to your HF model repo ID, e.g.:
  HF_MODEL_REPO=swaggersamantaray55/brain-tumor-ai-weights

If the weights already exist locally (e.g., baked into the Space via Git LFS),
this script exits immediately without downloading anything.

Can be used as a module (import download_models) OR run directly:
  python download_models.py
"""

import os
import sys

CLASSIFIER_PATH = "models/classifier/brain_tumor_classifier_best.pth"
SEGMENTER_PATH  = "models/segmenter/brain_tumor_segmenter_best.pth"


def models_exist() -> bool:
    return os.path.exists(CLASSIFIER_PATH) and os.path.exists(SEGMENTER_PATH)


def download_from_hub(repo_id: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download_models] huggingface_hub not installed — skipping download.")
        return

    os.makedirs("models/classifier", exist_ok=True)
    os.makedirs("models/segmenter",  exist_ok=True)

    files = {
        "brain_tumor_classifier_best.pth": CLASSIFIER_PATH,
        "brain_tumor_segmenter_best.pth":  SEGMENTER_PATH,
    }

    for filename, local_path in files.items():
        if os.path.exists(local_path):
            print(f"[download_models] ✔ Already exists: {local_path}")
            continue
        print(f"[download_models] ⬇  Downloading {filename} from {repo_id}...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=".",
                local_dir_use_symlinks=False,
            )
            # Move to expected path if hf_hub placed it elsewhere
            if os.path.abspath(downloaded) != os.path.abspath(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                os.replace(downloaded, local_path)
            print(f"[download_models] ✔ Saved to {local_path}")
        except Exception as e:
            print(f"[download_models] ✗ Failed to download {filename}: {e}")
            sys.exit(1)


def ensure_models() -> None:
    """Download model weights if not already present. Safe to call multiple times."""
    if models_exist():
        print("[download_models] ✔ Model weights already present — skipping download.")
        return

    repo_id = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo_id:
        print(
            "[download_models] ⚠  HF_MODEL_REPO env var not set.\n"
            "   Set it to your HF model repo (e.g. swaggersamantaray55/brain-tumor-ai-weights)\n"
            "   or upload weights to the Space via Git LFS.\n"
            "   Proceeding without weights — inference will fail until models are present."
        )
        return

    download_from_hub(repo_id)
    print("[download_models] ✔ All weights ready.")


# ─── Run on import (so `import download_models` triggers the download) ─────────
ensure_models()


if __name__ == "__main__":
    # Also callable directly: python download_models.py
    pass
