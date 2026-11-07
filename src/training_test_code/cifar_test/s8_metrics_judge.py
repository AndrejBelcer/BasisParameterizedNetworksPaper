import os
import csv
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4
import numpy as np

from src.training_test_code.cifar_test.s4_prepare_metrics import (
    cache_real_statistics, master_score_generator, TorchvisionKIDEvaluator,
    get_image_manifest, get_metric_protocol, score_dictionary_to_string, METRIC_LABELS,
)
from src.core.globals import GPU_CONFIG
from src.utilities.print_helper import PrintFields, PrintStatus, PrintSummary


def judgement_hash(evaluation: Dict[str, Any]) -> str:
    """A stable identity for the images, reference set and metric procedure."""
    text = json.dumps(evaluation, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cached_judgement(result_path: str, evaluation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ignore incomplete, outdated or damaged result files and recompute them."""
    if not os.path.isfile(result_path):
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved["evaluation"] != evaluation or saved["judgement_sha256"] != judgement_hash(evaluation):
            return None
        results = saved["results"]
        if results["Num_Images"] != evaluation["generated"]["image_count"]:
            return None
        if results["Folder"] != os.path.basename(evaluation["generated"]["folder"]):
            return None
        metric_keys = list(METRIC_LABELS) + ["Fidelity_IS_std", "Fidelity_KID_std_x1k"]
        if not all(type(results[key]) in (int, float) and np.isfinite(results[key]) for key in metric_keys):
            return None
        return results
    except (OSError, ValueError, KeyError, TypeError):
        print(f"[CACHE] Could not use {result_path}; recomputing this judgement.")
        return None


def write_result_file(path: str, text: str):
    """Replace a result only after the complete UTF-8 file has been written."""
    temporary_path = f"{path}.{uuid4().hex}.tmp"
    try:
        with open(temporary_path, "x", encoding="utf-8") as f:
            f.write(text)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def judgement_report(report: str, evaluation: Dict[str, Any]) -> str:
    reference = evaluation["reference"]
    protocol = evaluation["protocol"]
    text = f"Reference folder: {reference['folder']}\n"
    text += f"Reference images: {reference['image_count']:,}\n"
    text += f"Reference image SHA-256: {reference['images_sha256']}\n"
    text += f"Generated image SHA-256: {evaluation['generated']['images_sha256']}\n"
    text += f"Selection: {'first 50,000 sorted images (or all if fewer)' if evaluation['take_first_50k'] else 'all images'}\n"
    text += f"Judgement SHA-256: {judgement_hash(evaluation)}\n"
    text += "Package versions: " + ", ".join(f"{name}={value}" for name, value in protocol["package_versions"].items()) + "\n\n"
    return text + report + "\n"


def save_judgement(result_path: str, report_path: str, evaluation: Dict[str, Any], results: Dict[str, Any], report: str):
    saved = {
        "judgement_sha256": judgement_hash(evaluation),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation,
        "results": results,
    }
    # JSON is written last: only completed evaluations are reusable.
    write_result_file(report_path, judgement_report(report, evaluation))
    write_result_file(result_path, json.dumps(saved, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    PrintStatus(f"[SAVED] Readable report: {report_path}", "success")
    PrintStatus(f"[SAVED] Reusable results: {result_path}", "success")


def aggregate_and_save_results(results: List[Dict[str, Any]], output_csv: str):
    """
    Saves results to CSV. If multiple folders are evaluated, computes sample 
    Mean and sample Standard Deviation (ddof=1) across the runs.
    """
    if not results:
        PrintStatus("[WARNING] No results to save.", "warning")
        return

    # Column ordering
    fieldnames = [
        "Folder",
        "Num_Images",
        "PyTorch_FID",
        "Fidelity_FID",
        "Fidelity_IS_mean",
        "Fidelity_IS_std",
        "Fidelity_KID_mean_x1k",
        "Fidelity_KID_std_x1k",
        "Clean_FID",
        "Clean_Legacy_FID",
        "Clean_KID_x1k",
        "Clean_Legacy_KID_x1k",
        "Torchvision_KID_x1k",
    ]

    export_rows = list(results)

    # Compute statistical aggregation only when multiple folders are evaluated
    if len(results) > 1:
        metric_keys = [k for k in fieldnames if k not in ("Folder", "Num_Images")]
        
        mean_row: Dict[str, Any] = {
            "Folder": "MEAN",
            "Num_Images": int(np.round(np.mean([r["Num_Images"] for r in results]))),
        }
        std_row: Dict[str, Any] = {
            "Folder": "STD",
            "Num_Images": "",
        }

        for key in metric_keys:
            vals = [float(row[key]) for row in results]
            mean_row[key] = float(np.mean(vals))
            # Sample standard deviation (ddof=1) for academic reporting
            std_row[key] = float(np.std(vals, ddof=1))

        export_rows.append(mean_row)
        export_rows.append(std_row)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in export_rows:
            formatted = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float):
                    formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)

    PrintStatus(f"\n[SUCCESS] Final report saved to: {output_csv}", "success")


def main_judger(
    gen_folders: List[str],
    ref_folder: str,
    cache_dir: str,
    output_csv: str,
    device: str,
    take_first_50k: bool = False,
    force_recompute: bool = False,
):
    """
    Main orchestration function.
    """
    PrintSummary("Starting Generative Metrics Judger")
    
    # Hash image contents, not directory timestamps: saving reports must not
    # invalidate a judgement, while replacing images must invalidate it.
    print("[CACHE] Identifying reference images and evaluator versions...")
    reference = get_image_manifest(ref_folder)
    protocol = get_metric_protocol(device)
    reference_cache_identity = judgement_hash(dict(reference=reference, protocol=protocol))
    cache_info = None
    torchvision_evaluator = None

    all_results = []

    # 2. Evaluate generated folder(s)
    for gen_folder in gen_folders:
        if not os.path.isdir(gen_folder):
            PrintStatus(f"\n[WARNING] Folder '{gen_folder}' does not exist. Skipping.", "warning")
            continue

        print(f"\n[CACHE] Identifying generated images in: {gen_folder}")
        generated = get_image_manifest(gen_folder, take_first_50k)
        evaluation = {
            "format_version": 1,
            "generated": generated,
            "reference": reference,
            "protocol": protocol,
            "take_first_50k": take_first_50k,
        }
        # Distinct references, images or protocols can coexist in this folder.
        result_stem = os.path.join(gen_folder, f"metrics_{judgement_hash(evaluation)[:24]}")
        result_path = result_stem + ".json"
        report_path = result_stem + ".txt"
        res = None if force_recompute else load_cached_judgement(result_path, evaluation)
        if res is not None:
            print(f"[CACHE HIT] Loaded judgement: {result_path}")
            report = score_dictionary_to_string(res["Folder"], res["Num_Images"], res)
            if not os.path.isfile(report_path):
                write_result_file(report_path, judgement_report(report, evaluation))
        else:
            print("[CACHE MISS] Computing metrics..." if not force_recompute else "[CACHE] Recomputing metrics...")
            # All-cache-hit runs never load models or extract reference features.
            if cache_info is None:
                cache_info = cache_real_statistics(ref_folder, cache_dir, device, cache_identity=reference_cache_identity)
                torchvision_evaluator = TorchvisionKIDEvaluator(ref_folder, cache_info["torchvision_cache"], device)
            res, report = master_score_generator(gen_folder=gen_folder, ref_folder=ref_folder, cache_info=cache_info, device=device, take_first_50k=take_first_50k, torchvision_evaluator=torchvision_evaluator)
            save_judgement(result_path, report_path, evaluation, res, report)
        all_results.append(res)

    if not all_results:
        PrintStatus("[ERROR] No valid folders evaluated.", "error")
        return

    # 3. Save report
    aggregate_and_save_results(all_results, output_csv)

    # 4. Final summary
    if len(all_results) > 1:
        PrintSummary("FINAL MULTI-RUN STATISTICAL SUMMARY (Mean ± Sample Std [ddof=1])")
        summary_rows = []
        for key, label in METRIC_LABELS.items():
            if key.endswith("_x1k"):
                label += " (x10³)"
            vals = [r[key] for r in all_results]
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1)
            summary_rows.append((label, f"{mean_v:.4f} ± {std_v:.4f}"))
        PrintFields(summary_rows)
    return all_results


def MakeJudgement(
    output_csv: str = "final_metrics_report",
    gen_folders: Optional[List[str]] = None,
    ref_folder: str = "data/cifar_official/fid_reference_flat",
    take_first_50k: bool = False,
    force_recompute: bool = False,
):
    """
    Entry-point for running metrics judgment.
    
    Args:
        output_csv: Name of CSV file (stored in outputs/benchmarks/).
        gen_folders: List of folder paths to evaluate. If None, defaults to the 5 seed runs.
        ref_folder: Reference image folder; its path and contents identify the reference set.
        take_first_50k: If True and a folder has > 50k images (e.g. 51,200),
                        evaluates only the first 50,000 images for benchmark parity.
        force_recompute: Recalculate scores even if matching per-folder JSON results exist.
                         Reference feature/statistics caches can still be reused.
    """
    DEVICE = GPU_CONFIG.device
    CACHE_DIR = os.path.join("cache", "metrics")
    OUTPUT_CSV = f"outputs/benchmarks/{output_csv}.csv"

    if gen_folders is None:
        gen_folders = [
            "outputs/benchmarks/full_evaluations/CifEModel_S66_50000_s11_f1.0",
            "outputs/benchmarks/full_evaluations/CifEModel_S66_50000_s12_f1.0",
            "outputs/benchmarks/full_evaluations/CifEModel_S66_50000_s13_f1.0",
            "outputs/benchmarks/full_evaluations/CifEModel_S66_50000_s14_f1.0",
            "outputs/benchmarks/full_evaluations/CifEModel_S66_50000_s15_f1.0",
        ]

    return main_judger(
        gen_folders=gen_folders,
        ref_folder=ref_folder,
        cache_dir=CACHE_DIR,
        output_csv=OUTPUT_CSV,
        device=DEVICE,
        take_first_50k=take_first_50k,
        force_recompute=force_recompute,
    )
