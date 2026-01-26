"""Test script to verify all imports and cross-platform compatibility.

Usage:
    python test_imports.py

This script checks:
- All module imports work correctly
- Pathlib cross-platform compatibility
- Script imports work from different directories
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    errors = []
    
    try:
        from src.datasets.div2k import make_div2k_loaders, DIV2KDataset
        print("[OK] src.datasets.div2k")
    except ImportError as e:
        print(f"[FAIL] src.datasets.div2k - {e}")
        errors.append("src.datasets.div2k")
    
    try:
        from src.config import load_config
        print("[OK] src.config")
    except ImportError as e:
        print(f"[FAIL] src.config - {e}")
        errors.append("src.config")
    
    try:
        from src.utils.metrics import psnr, ssim
        print("[OK] src.utils.metrics")
    except ImportError as e:
        print(f"[FAIL] src.utils.metrics - {e}")
        errors.append("src.utils.metrics")
    
    try:
        from src.utils.device import get_device
        print("[OK] src.utils.device")
    except ImportError as e:
        print(f"[FAIL] src.utils.device - {e}")
        errors.append("src.utils.device")
    
    try:
        from src.utils.seed import set_seed
        print("[OK] src.utils.seed")
    except ImportError as e:
        print(f"[FAIL] src.utils.seed - {e}")
        errors.append("src.utils.seed")
    
    return errors


def test_pathlib():
    """Test pathlib cross-platform compatibility."""
    print("\n" + "=" * 60)
    print("Testing pathlib cross-platform compatibility...")
    print("=" * 60)
    
    try:
        test_path = Path("data/raw/DIV2K") / "DIV2K_train_HR"
        print(f"[OK] Path joining: {test_path}")
        
        resolved = Path(__file__).resolve()
        print(f"[OK] Path.resolve(): {resolved}")
        
        parent = Path(__file__).parent
        print(f"[OK] Path.parent: {parent}")
        
        parts = test_path.parts
        print(f"[OK] Path.parts: {parts}")
        
        return True
    except Exception as e:
        print(f"[FAIL] pathlib test - {e}")
        return False


def test_config_loading():
    """Test config loading."""
    print("\n" + "=" * 60)
    print("Testing config loading...")
    print("=" * 60)
    
    try:
        from src.config import load_config
        
        config_path = Path("configs/common.yaml")
        if config_path.exists():
            cfg = load_config(config_path)
            print(f"[OK] Loaded configs/common.yaml")
            print(f"     Project name: {cfg.get('project', {}).get('name', 'N/A')}")
            return True
        else:
            print(f"[SKIP] configs/common.yaml not found (expected if data not present)")
            return True
    except Exception as e:
        print(f"[FAIL] Config loading - {e}")
        return False


def test_device():
    """Test device selection."""
    print("\n" + "=" * 60)
    print("Testing device selection...")
    print("=" * 60)
    
    try:
        from src.utils.device import get_device
        device = get_device()
        print(f"[OK] Device selected: {device}")
        
        import torch
        if device.type == "cuda":
            print(f"     CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"     GPU: {torch.cuda.get_device_name(0)}")
        elif device.type == "mps":
            print(f"     MPS available: {torch.backends.mps.is_available()}")
        else:
            print(f"     Using CPU")
        
        return True
    except Exception as e:
        print(f"[FAIL] Device selection - {e}")
        return False


def test_metrics():
    """Test metrics functions."""
    print("\n" + "=" * 60)
    print("Testing metrics functions...")
    print("=" * 60)
    
    try:
        import torch
        from src.utils.metrics import psnr, ssim
        
        img1 = torch.rand(3, 64, 64)
        img2 = img1.clone()
        
        psnr_val = psnr(img1, img2)
        ssim_val = ssim(img1, img2)
        
        print(f"[OK] PSNR for identical images: {psnr_val}")
        print(f"[OK] SSIM for identical images: {ssim_val:.6f}")
        
        img3 = torch.rand(3, 64, 64)
        psnr_val2 = psnr(img1, img3)
        ssim_val2 = ssim(img1, img3)
        
        print(f"[OK] PSNR for different images: {psnr_val2:.4f} dB")
        print(f"[OK] SSIM for different images: {ssim_val2:.6f}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Metrics test - {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Cross-platform Import and Compatibility Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Project root: {project_root}")
    print()
    
    all_passed = True
    
    import_errors = test_imports()
    if import_errors:
        all_passed = False
        print(f"\n[ERROR] Failed imports: {', '.join(import_errors)}")
    
    pathlib_ok = test_pathlib()
    if not pathlib_ok:
        all_passed = False
    
    config_ok = test_config_loading()
    if not config_ok:
        all_passed = False
    
    device_ok = test_device()
    if not device_ok:
        all_passed = False
    
    metrics_ok = test_metrics()
    if not metrics_ok:
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed and not import_errors:
        print("RESULT: All tests PASSED!")
        print("=" * 60)
        return 0
    else:
        print("RESULT: Some tests FAILED!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

