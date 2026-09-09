"""
Download the MONAI BraTS segmentation bundle directly using urllib.
No torch/MPS initialization — just HTTP download.
"""
import os, sys, zipfile, urllib.request, shutil
from pathlib import Path

BUNDLE_NAME    = "brats_mri_segmentation"
BUNDLE_VERSION = "0.4.2"
BUNDLE_URL     = (
    f"https://api.ngc.nvidia.com/v2/models/nvidia/monai/{BUNDLE_NAME}/"
    f"versions/{BUNDLE_VERSION}/zip"
)
# Fallback: MONAI GitHub releases
FALLBACK_URL = (
    f"https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/"
    f"{BUNDLE_NAME}_v{BUNDLE_VERSION}.zip"
)

DEST = Path(__file__).parent.parent / "models" / "monai"
ZIP_PATH = DEST / f"{BUNDLE_NAME}.zip"

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {pct:.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  {mb:.1f} MB downloaded...", end="", flush=True)

def download_bundle():
    DEST.mkdir(parents=True, exist_ok=True)
    bundle_dir = DEST / BUNDLE_NAME

    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        print(f"✓ Bundle already cached at: {bundle_dir}")
        list_files(bundle_dir)
        return bundle_dir

    print(f"Downloading MONAI BraTS bundle v{BUNDLE_VERSION}...")
    print(f"Destination: {ZIP_PATH}")
    print()

    # Try primary URL
    for url in [BUNDLE_URL, FALLBACK_URL]:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, ZIP_PATH, reporthook=show_progress)
            print("\n✓ Download complete")
            break
        except Exception as e:
            print(f"\n  Failed: {e}")
            if ZIP_PATH.exists():
                ZIP_PATH.unlink()
    else:
        # Both URLs failed — try monai python API without MPS
        print("\nDirect URLs failed. Trying MONAI API (CPU only)...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Prevent MPS from initializing
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             f"import os; os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1';"
             f"from monai.bundle import download; "
             f"download(name='{BUNDLE_NAME}', version='{BUNDLE_VERSION}', "
             f"bundle_dir='{DEST}', source='monaihosting')"],
            capture_output=False, text=True
        )
        if result.returncode == 0:
            print("✓ Download via MONAI API complete")
            bundle_dir = DEST / BUNDLE_NAME
            list_files(bundle_dir)
            return bundle_dir
        else:
            print("✗ All download methods failed.")
            sys.exit(1)

    # Unzip
    print(f"\nExtracting to {DEST / BUNDLE_NAME}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        # List contents to find the bundle root
        names = zf.namelist()
        root_dirs = set(n.split('/')[0] for n in names if '/' in n)
        print(f"  Archive root(s): {root_dirs}")
        zf.extractall(DEST)
    
    # Clean up zip
    ZIP_PATH.unlink()
    
    # Rename if needed (bundle dir might be named differently)
    extracted = DEST / BUNDLE_NAME
    if not extracted.exists():
        # Find what was extracted
        for d in DEST.iterdir():
            if d.is_dir() and d.name != BUNDLE_NAME:
                d.rename(extracted)
                print(f"  Renamed {d.name} → {BUNDLE_NAME}")
                break

    print(f"✓ Bundle extracted to: {extracted}")
    list_files(extracted)
    return extracted

def list_files(bundle_dir):
    print(f"\nBundle contents:")
    for f in sorted(Path(bundle_dir).rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(bundle_dir)}  ({size_mb:.2f} MB)")

if __name__ == "__main__":
    bundle = download_bundle()
    print(f"\n✓ Ready. Bundle at: {bundle}")
    print("  Restart flask_app_3d.py and click 'Run Demo Mode'")
