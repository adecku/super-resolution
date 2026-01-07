"""Collect results from multiple metrics.json files into a single CSV and JSON file."""

import json
import csv
import sys
import re
from pathlib import Path

# Add project root to path so imports work when running directly
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def extract_model_from_path(path: Path) -> str:
    """Extract model name from path like 'eval_srcnn_x2' or 'bicubic_baseline_x2'."""
    # Get parent directory name
    parent_name = path.parent.name
    
    # Handle eval_srcnn_x2, eval_edsr_x2, etc.
    if parent_name.startswith("eval_"):
        model = parent_name.replace("eval_", "").split("_x")[0]
        return model
    
    # Handle bicubic_baseline_x2
    if "bicubic" in parent_name.lower() or "baseline" in parent_name.lower():
        return "bicubic"
    
    # Default: try to extract from filename or parent
    if "srcnn" in parent_name.lower():
        return "srcnn"
    if "edsr" in parent_name.lower():
        return "edsr"
    if "swinir" in parent_name.lower():
        return "swinir"
    
    return "unknown"


def extract_epoch_from_checkpoint(checkpoint_path: str) -> int | None:
    """Extract epoch number from checkpoint path like 'ckpt_ep9.pth'."""
    if not checkpoint_path:
        return None
    
    match = re.search(r"ckpt_ep(\d+)\.pth", checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def extract_run_name_from_checkpoint(checkpoint_path: str) -> str | None:
    """Extract run name from checkpoint path like 'outputs/runs/srcnn_x2/20260107-144217/ckpt_ep9.pth'."""
    if not checkpoint_path:
        return None
    
    # Extract timestamp directory name
    match = re.search(r"runs/[^/]+/([^/]+)/", checkpoint_path)
    if match:
        return match.group(1)
    return None


def collect_metrics_files(figures_dir: Path) -> list[dict]:
    """Collect all metrics.json files and extract data."""
    results = []
    
    # Find all metrics.json files
    metrics_files = list(figures_dir.rglob("metrics.json"))
    
    # Also check for bicubic_baseline files
    baseline_files = list(figures_dir.glob("bicubic_baseline_x*.json"))
    
    all_files = metrics_files + baseline_files
    
    for json_file in all_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract model name from path
            model = extract_model_from_path(json_file)
            
            # Extract scale
            scale = data.get("scale")
            if scale is None:
                # Try to extract from filename
                match = re.search(r"_x(\d+)", json_file.stem)
                if match:
                    scale = int(match.group(1))
            
            # Extract metrics
            psnr = data.get("psnr")
            ssim = data.get("ssim")
            
            # Extract optional fields
            epoch = None
            run_name = None
            checkpoint = data.get("checkpoint", "")
            
            if checkpoint:
                epoch = extract_epoch_from_checkpoint(checkpoint)
                run_name = extract_run_name_from_checkpoint(checkpoint)
            
            # Create result entry
            entry = {
                "model": model,
                "scale": scale,
                "psnr": psnr,
                "ssim": ssim,
            }
            
            # Add optional fields if available
            if epoch is not None:
                entry["epoch"] = epoch
            if run_name:
                entry["run_name"] = run_name
            if checkpoint:
                entry["checkpoint"] = checkpoint
            
            # Add source file for reference
            entry["source_file"] = str(json_file.relative_to(figures_dir))
            
            results.append(entry)
            
        except FileNotFoundError:
            # Skip missing files
            continue
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue
    
    return results


def save_results_csv(results: list[dict], output_file: Path):
    """Save results to CSV file."""
    if not results:
        print("No results to save to CSV")
        return
    
    # Get all possible keys
    all_keys = set()
    for entry in results:
        all_keys.update(entry.keys())
    
    # Define column order
    column_order = ["model", "scale", "psnr", "ssim", "epoch", "run_name", "checkpoint", "source_file"]
    # Add any other keys that weren't in the predefined order
    for key in sorted(all_keys):
        if key not in column_order:
            column_order.append(key)
    
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=column_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def save_results_json(results: list[dict], output_file: Path):
    """Save results to JSON file."""
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    figures_dir = Path("outputs/figures")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not figures_dir.exists():
        print(f"Figures directory not found: {figures_dir}")
        print("No results to collect.")
        return
    
    print(f"Collecting results from: {figures_dir}")
    
    # Collect all metrics
    results = collect_metrics_files(figures_dir)
    
    if not results:
        print("No metrics files found.")
        return
    
    # Sort results by model, then scale, then epoch (if available)
    results.sort(key=lambda x: (
        x.get("model", ""),
        x.get("scale", 0),
        x.get("epoch", -1) if x.get("epoch") is not None else -1
    ))
    
    # Save to CSV
    csv_file = output_dir / "results.csv"
    save_results_csv(results, csv_file)
    print(f"Saved {len(results)} results to: {csv_file}")
    
    # Save to JSON
    json_file = output_dir / "results.json"
    save_results_json(results, json_file)
    print(f"Saved {len(results)} results to: {json_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total entries: {len(results)}")
    
    # Group by model
    by_model = {}
    for entry in results:
        model = entry.get("model", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(entry)
    
    for model, entries in sorted(by_model.items()):
        print(f"  {model}: {len(entries)} entries")
        # Show scales
        scales = sorted(set(e.get("scale") for e in entries if e.get("scale") is not None))
        if scales:
            print(f"    Scales: {', '.join(f'x{s}' for s in scales)}")


if __name__ == "__main__":
    main()

