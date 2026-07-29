"""
Download model weights from Hugging Face Hub at container startup.

Set the environment variable HF_MODEL_REPO to your HF model repo ID, e.g.:
  HF_MODEL_REPO=swaggersamantaray55/brain-tumor-ai-weights
"""

import os
import sys
import shutil

CLASSIFIER_PATH = "models/classifier/brain_tumor_classifier_best.pth"
SEGMENTER_PATH  = "models/segmenter/brain_tumor_segmenter_best.pth"

# Use absolute paths to avoid CWD confusion
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_ABS = os.path.join(_BASE_DIR, CLASSIFIER_PATH)
SEGMENTER_ABS  = os.path.join(_BASE_DIR, SEGMENTER_PATH)


def models_exist() -> bool:
    clf = os.path.exists(CLASSIFIER_ABS)
    seg = os.path.exists(SEGMENTER_ABS)
    print(f"[download_models] Checking models:")
    print(f"  classifier ({CLASSIFIER_ABS}): {'✔' if clf else '✗'}")
    print(f"  segmenter  ({SEGMENTER_ABS}): {'✔' if seg else '✗'}")
    return clf and seg


def download_one(repo_id: str, filename: str, dest_abs: str) -> None:
    """Download a single file from HF Hub directly into dest_abs."""
    if os.path.exists(dest_abs):
        print(f"[download_models] ✔ Already exists: {dest_abs}")
        return

    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    print(f"[download_models] ⬇  Downloading {filename} → {dest_abs}")

    try:
        from huggingface_hub import hf_hub_download

        # Download directly into the correct directory
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(dest_abs),
            local_dir_use_symlinks=False,
        )
        print(f"[download_models]    hf_hub returned: {downloaded}")

        # Ensure it's at the exact expected path
        if os.path.abspath(downloaded) != os.path.abspath(dest_abs):
            print(f"[download_models]    Moving {downloaded} → {dest_abs}")
            shutil.move(downloaded, dest_abs)

        if os.path.exists(dest_abs):
            size_mb = os.path.getsize(dest_abs) / 1e6
            print(f"[download_models] ✔ Saved ({size_mb:.1f} MB): {dest_abs}")
        else:
            raise FileNotFoundError(f"File missing after download: {dest_abs}")

    except Exception as e:
        print(f"[download_models] ✗ hf_hub_download failed: {e}")
        # Fallback: direct HTTP download
        _http_download(repo_id, filename, dest_abs)


def _http_download(repo_id: str, filename: str, dest_abs: str) -> None:
    """Fallback: download via HTTP from HF Hub public URL."""
    import urllib.request
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"[download_models] ↪  HTTP fallback: {url}")
    try:
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        urllib.request.urlretrieve(url, dest_abs)
        size_mb = os.path.getsize(dest_abs) / 1e6
        print(f"[download_models] ✔ HTTP download done ({size_mb:.1f} MB): {dest_abs}")
    except Exception as e2:
        print(f"[download_models] ✗ HTTP fallback also failed: {e2}")
        sys.exit(1)


def ensure_models() -> None:
    """Download model weights if not already present. Called at module import."""
    print(f"[download_models] Working dir: {os.getcwd()}")
    print(f"[download_models] Script dir:  {_BASE_DIR}")

    if models_exist():
        print("[download_models] ✔ All weights present — skipping download.")
        return

    repo_id = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo_id:
        print(
            "[download_models] ⚠  HF_MODEL_REPO not set — inference will fail.\n"
            "   Set HF_MODEL_REPO=swaggersamantaray55/brain-tumor-ai-weights in Space settings."
        )
        return

    print(f"[download_models] Repo: {repo_id}")
    download_one(repo_id, "brain_tumor_classifier_best.pth", CLASSIFIER_ABS)
    download_one(repo_id, "brain_tumor_segmenter_best.pth",  SEGMENTER_ABS)
    print("[download_models] ✔ All weights ready.")


# ─── Auto-run on import ────────────────────────────────────────────────────────
ensure_models()


if __name__ == "__main__":
    pass
