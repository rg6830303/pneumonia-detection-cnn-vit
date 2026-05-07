"""
Streamlit dashboard for the pneumonia detection CNN/ViT repository.

Run locally with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import torch
except Exception:
    torch = None

import config

try:
    from show_all_results import display_all_results
except Exception:
    display_all_results = None


PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_DIR = Path(config.DATA_DIR)
RESULTS_DIR = Path(config.RESULTS_DIR)
MODELS_DIR = Path(config.MODELS_DIR)
EVALUATION_DIR = RESULTS_DIR / "evaluation"
TRAINING_PLOTS_DIR = RESULTS_DIR / "training_plots"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODEL_EXTENSIONS = {".pth", ".pt", ".ckpt"}
MODEL_TYPES = ["hybrid", "cnn_only", "vit_only"]
SPLITS = ["train", "val", "test"]
CLASS_NAMES = list(getattr(config, "CLASS_NAMES", ["NORMAL", "PNEUMONIA"]))


st.set_page_config(
    page_title="Pneumonia Detection Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


class StreamlitLogBuffer(io.StringIO):
    """Capture stdout/stderr while periodically updating a Streamlit placeholder."""

    def __init__(self, placeholder, max_chars: int = 14000):
        super().__init__()
        self.placeholder = placeholder
        self.max_chars = max_chars
        self._last_render = 0.0

    def write(self, text: str) -> int:
        result = super().write(text)
        now = time.time()
        if "\n" in text or now - self._last_render > 0.35:
            self.render()
            self._last_render = now
        return result

    def render(self) -> None:
        value = self.getvalue()
        tail = value[-self.max_chars:] if len(value) > self.max_chars else value
        self.placeholder.code(tail or "Waiting for output...", language="text")


@contextlib.contextmanager
def temporary_config(**updates: Any):
    missing = object()
    old_values = {name: getattr(config, name, missing) for name in updates}
    try:
        for name, value in updates.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in old_values.items():
            if value is missing:
                delattr(config, name)
            else:
                setattr(config, name, value)


def run_with_live_logs(func, *args, log_placeholder=None, **kwargs) -> Tuple[Any, str]:
    placeholder = log_placeholder or st.empty()
    buffer = StreamlitLogBuffer(placeholder)
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = func(*args, **kwargs)
    buffer.render()
    return result, buffer.getvalue()


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def format_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return ""


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def count_images(path: Path) -> int:
    return len(image_files(path))


@st.cache_data(show_spinner=False)
def inspect_expected_dataset(root: str) -> Dict[str, Any]:
    root_path = Path(root)
    rows = []
    missing = []
    totals = {name: 0 for name in CLASS_NAMES}
    flat_has_classes = all((root_path / name).is_dir() for name in CLASS_NAMES)
    split_has_classes = all(
        (root_path / split / name).is_dir()
        for split in SPLITS
        for name in CLASS_NAMES
    )

    if root_path.exists() and flat_has_classes and not split_has_classes:
        ratios = getattr(
            config,
            "DATA_SPLIT_RATIOS",
            {"train": 0.70, "val": 0.15, "test": 0.15},
        )
        split_counts = {split: {class_name: 0 for class_name in CLASS_NAMES} for split in SPLITS}
        for class_name in CLASS_NAMES:
            count = count_images(root_path / class_name)
            train_count = int(count * ratios.get("train", 0.70))
            val_count = int(count * ratios.get("val", 0.15))
            test_count = count - train_count - val_count
            split_counts["train"][class_name] = train_count
            split_counts["val"][class_name] = val_count
            split_counts["test"][class_name] = test_count
            totals[class_name] = count

        for split in SPLITS:
            row = {"split": split, **split_counts[split]}
            row["total"] = sum(row[name] for name in CLASS_NAMES)
            row["complete"] = True
            rows.append(row)

        train_normal = rows[0].get("NORMAL", 0)
        train_pneumonia = rows[0].get("PNEUMONIA", 0)
        pos_weight = train_normal / train_pneumonia if train_pneumonia else None
        return {
            "root_exists": True,
            "complete": True,
            "layout": "flat",
            "rows": rows,
            "missing": [],
            "totals": totals,
            "pos_weight": pos_weight,
        }

    for split in SPLITS:
        row: Dict[str, Any] = {"split": split}
        split_complete = True
        for class_name in CLASS_NAMES:
            class_dir = root_path / split / class_name
            exists = class_dir.is_dir()
            if not exists:
                missing.append(str(class_dir))
                split_complete = False
            count = count_images(class_dir) if exists else 0
            row[class_name] = count
            totals[class_name] += count
        row["total"] = sum(row[name] for name in CLASS_NAMES)
        row["complete"] = split_complete
        rows.append(row)

    train_normal = rows[0].get("NORMAL", 0)
    train_pneumonia = rows[0].get("PNEUMONIA", 0)
    pos_weight = train_normal / train_pneumonia if train_pneumonia else None

    return {
        "root_exists": root_path.exists(),
        "complete": root_path.exists() and not missing,
        "layout": "split" if root_path.exists() and not missing else "missing",
        "rows": rows,
        "missing": missing,
        "totals": totals,
        "pos_weight": pos_weight,
    }


@st.cache_data(show_spinner=False)
def inspect_flat_dataset(root: str) -> Dict[str, Any]:
    root_path = Path(root)
    rows = []
    totals = {name: 0 for name in CLASS_NAMES}
    for class_name in CLASS_NAMES:
        class_dir = root_path / class_name
        count = count_images(class_dir)
        totals[class_name] = count
        rows.append({"class": class_name, "path": str(class_dir), "images": count})
    return {
        "root_exists": root_path.exists(),
        "has_classes": all((root_path / name).is_dir() for name in CLASS_NAMES),
        "rows": rows,
        "totals": totals,
        "total": sum(totals.values()),
    }


def alternate_dataset_roots() -> List[Path]:
    candidates = [
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data" / "chest_xray",
        PROJECT_ROOT / "data",
    ]
    seen = set()
    roots = []
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen or candidate.resolve() == DATA_DIR.resolve():
            continue
        seen.add(resolved)
        if candidate.exists():
            roots.append(candidate)
    return roots


def best_available_image_folder() -> Optional[Path]:
    expected = inspect_expected_dataset(str(DATA_DIR))
    if expected["complete"]:
        if expected.get("layout") == "flat":
            for class_name in CLASS_NAMES:
                folder = DATA_DIR / class_name
                if count_images(folder):
                    return folder
        for split in SPLITS:
            for class_name in CLASS_NAMES:
                folder = DATA_DIR / split / class_name
                if count_images(folder):
                    return folder
    for root in alternate_dataset_roots():
        flat = inspect_flat_dataset(str(root))
        if flat["has_classes"]:
            for class_name in CLASS_NAMES:
                folder = root / class_name
                if count_images(folder):
                    return folder
    return None


def default_image_for_class(class_name: str) -> str:
    expected = inspect_expected_dataset(str(DATA_DIR))
    if expected["complete"]:
        if expected.get("layout") == "flat":
            files = image_files(DATA_DIR / class_name)
            if files:
                return str(files[0])
        for split in ["test", "val", "train"]:
            files = image_files(DATA_DIR / split / class_name)
            if files:
                return str(files[0])
    for root in alternate_dataset_roots():
        files = image_files(root / class_name)
        if files:
            return str(files[0])
    return ""


def denormalize_tensor(image_tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std) + mean
    return np.clip(image, 0, 1)


def load_training_loader_preview(batch_size: int = 4) -> Tuple[Optional[np.ndarray], Optional[str], str]:
    expected = inspect_expected_dataset(str(DATA_DIR))
    if expected["complete"]:
        from dataset import get_data_loaders

        train_loader, _, _ = get_data_loaders(
            data_dir=str(DATA_DIR),
            batch_size=max(1, min(batch_size, 8)),
            num_workers=0,
            use_augmentation=False,
        )
        images, labels = next(iter(train_loader))
        label_idx = int(labels[0].item())
        return denormalize_tensor(images[0]), CLASS_NAMES[label_idx], "training loader"

    folder = best_available_image_folder()
    if folder is None:
        return None, None, "no dataset image found"
    files = image_files(folder)
    if not files:
        return None, None, "no dataset image found"
    image = Image.open(files[0]).convert("RGB")
    label = folder.name if folder.name in CLASS_NAMES else "unknown"
    return np.array(image), label, f"fallback image from {rel_path(files[0])}"


def infer_model_type(path_or_name: Any) -> str:
    text = str(path_or_name).lower()
    if "cnn_only" in text or "ablation_cnn" in text:
        return "cnn_only"
    if "vit_only" in text or "ablation_vit" in text:
        return "vit_only"
    if "hybrid" in text:
        return "hybrid"
    return "hybrid"


@st.cache_data(show_spinner=False)
def list_model_folders(models_dir: str) -> List[Dict[str, Any]]:
    root = Path(models_dir)
    if not root.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        checkpoints = [
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
        ]
        history_path = folder / "training_history.json"
        metadata_files = [p for p in folder.glob("*.json") if p.name != "training_history.json"]
        rows.append(
            {
                "folder": str(folder),
                "model_type": infer_model_type(folder.name),
                "checkpoints": [str(p) for p in checkpoints],
                "checkpoint_count": len(checkpoints),
                "best_model": str(folder / "best_model.pth") if (folder / "best_model.pth").exists() else "",
                "final_model": str(folder / "final_model.pth") if (folder / "final_model.pth").exists() else "",
                "history": str(history_path) if history_path.exists() else "",
                "metadata_files": [str(p) for p in metadata_files],
                "modified": format_mtime(folder),
                "modified_ts": folder.stat().st_mtime,
            }
        )
    return rows


def list_checkpoints() -> List[Dict[str, Any]]:
    folders = list_model_folders(str(MODELS_DIR))
    checkpoints: List[Dict[str, Any]] = []
    for folder in folders:
        for checkpoint in folder["checkpoints"]:
            path = Path(checkpoint)
            checkpoints.append(
                {
                    "path": str(path),
                    "label": f"{infer_model_type(path)} | {rel_path(path)} | {format_mtime(path)}",
                    "model_type": infer_model_type(path),
                    "modified_ts": path.stat().st_mtime,
                }
            )
    checkpoints.sort(key=lambda item: item["modified_ts"], reverse=True)
    return checkpoints


def checkpoint_selector(label: str, key: str) -> Tuple[Optional[str], str]:
    checkpoints = list_checkpoints()
    selected_path = None
    inferred_type = "hybrid"

    if checkpoints:
        options = [item["label"] for item in checkpoints]
        selected = st.selectbox(label, options, key=f"{key}_select")
        selected_item = checkpoints[options.index(selected)]
        selected_path = selected_item["path"]
        inferred_type = selected_item["model_type"]
    else:
        st.warning("No .pth, .pt, or .ckpt checkpoint files were found under models/.")

    manual = st.text_input(
        "Or enter a checkpoint path",
        value="" if selected_path else "",
        key=f"{key}_manual",
        placeholder="models/hybrid_YYYYMMDD_HHMMSS/best_model.pth",
    ).strip()
    if manual:
        selected_path = manual
        inferred_type = infer_model_type(manual)
    return selected_path, inferred_type


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def json_to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                rows.append({"field": key, "value": json.dumps(value, indent=2)})
            else:
                rows.append({"field": key, "value": value})
        return pd.DataFrame(rows)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame([{"value": data}])


def render_metric_cards(metrics: Dict[str, Any]) -> None:
    ordered = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    cols = st.columns(len(ordered))
    for col, name in zip(cols, ordered):
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            col.metric(name.upper(), f"{value:.4f}")
        else:
            col.metric(name.upper(), "n/a")


def plot_class_balance(split_rows: List[Dict[str, Any]]) -> None:
    rows = []
    for row in split_rows:
        for class_name in CLASS_NAMES:
            rows.append({"split": row["split"], "class": class_name, "count": row.get(class_name, 0)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = df.pivot(index="split", columns="class", values="count").reindex(SPLITS)
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    ax.set_title("Class Balance by Split")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def display_image_grid(paths: Iterable[Path], max_images: int = 8) -> None:
    image_paths = [p for p in paths if p.exists() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        return
    cols = st.columns(min(4, len(image_paths[:max_images])))
    for idx, path in enumerate(image_paths[:max_images]):
        cols[idx % len(cols)].image(str(path), caption=rel_path(path), use_container_width=True)


def recent_artifacts(directory: Path, since_ts: Optional[float] = None) -> List[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.rglob("*") if p.is_file()]
    if since_ts is not None:
        files = [p for p in files if p.stat().st_mtime >= since_ts]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def save_uploaded_image(uploaded_file) -> Optional[Path]:
    if uploaded_file is None:
        return None
    upload_dir = RESULTS_DIR / "streamlit_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    with path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return path


def predict_single_image(image_path: str, checkpoint_path: str, model_type: str) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is not available in this environment.")

    from dataset import get_transforms
    from model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = get_transforms("test", use_augmentation=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor)).squeeze().detach().cpu().item())
    predicted_index = int(probability >= 0.5)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probability if predicted_index == 1 else 1.0 - probability
    return {
        "predicted_class": predicted_class,
        "pneumonia_probability": probability,
        "confidence": confidence,
    }


def extract_report_block(log_text: str) -> str:
    marker = "CLASSIFICATION REPORT"
    idx = log_text.find(marker)
    if idx == -1:
        return log_text[-4000:]
    return log_text[idx : idx + 5000]


def write_streamlit_evaluation_summary(result: Dict[str, Any], log_text: str) -> Path:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    model_type = result.get("model_type", "model")
    path = EVALUATION_DIR / f"{model_type}_streamlit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        **result,
        "classification_report_text": extract_report_block(log_text),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    return path


def run_repo_script(script_name: str, timeout: int = 45) -> str:
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += "\n--- STDERR ---\n" + completed.stderr
    if completed.returncode != 0:
        output += f"\n--- EXIT CODE: {completed.returncode} ---"
    return output


def latest_evaluation_json() -> Optional[Path]:
    if not EVALUATION_DIR.exists():
        return None
    json_files = sorted(EVALUATION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0] if json_files else None


def render_home() -> None:
    st.title("Pneumonia Detection Workbench")
    st.write(
        "A local Streamlit interface for inspecting the dataset, launching training, "
        "running evaluation, generating explainability outputs, and reviewing saved results."
    )
    st.caption("Created by Agnideep Ghorai")

    expected = inspect_expected_dataset(str(DATA_DIR))
    checkpoints = list_checkpoints()
    latest_eval = latest_evaluation_json()

    cols = st.columns(4)
    dataset_label = f"ready ({expected.get('layout', 'unknown')})" if expected["complete"] else "incomplete"
    cols[0].metric("Configured Dataset", dataset_label)
    cols[1].metric("Model Folders", len(list_model_folders(str(MODELS_DIR))))
    cols[2].metric("Checkpoints", len(checkpoints))
    cols[3].metric("Latest Eval", latest_eval.name if latest_eval else "none")

    st.subheader("Repository Paths")
    st.dataframe(
        pd.DataFrame(
            [
                {"name": "Project root", "path": str(PROJECT_ROOT)},
                {"name": "config.DATA_DIR", "path": str(DATA_DIR)},
                {"name": "config.MODELS_DIR", "path": str(MODELS_DIR)},
                {"name": "config.RESULTS_DIR", "path": str(RESULTS_DIR)},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not expected["complete"]:
        st.warning(
            "Open the Dataset tab for the exact missing paths and detected fallback data."
        )


def render_dataset() -> None:
    st.title("Dataset")
    st.write(f"Configured dataset root from `config.DATA_DIR`: `{DATA_DIR}`")

    expected = inspect_expected_dataset(str(DATA_DIR))
    if expected["complete"] and expected.get("layout") == "flat":
        st.success(
            "Flat dataset structure detected. The app will use dataset/NORMAL and dataset/PNEUMONIA "
            "with deterministic train/val/test splits."
        )
    elif expected["complete"]:
        st.success("Expected train/val/test dataset structure is complete.")
    elif not expected["root_exists"]:
        st.warning("Configured dataset root does not exist.")
    else:
        st.warning("Configured dataset root exists, but required split/class folders are missing.")

    split_df = pd.DataFrame(expected["rows"])
    st.subheader("Dataset Split Counts")
    st.dataframe(split_df, use_container_width=True, hide_index=True)

    metric_cols = st.columns(4)
    for idx, split in enumerate(SPLITS):
        total = int(split_df.loc[split_df["split"] == split, "total"].iloc[0])
        metric_cols[idx].metric(split.upper(), total)
    if expected["pos_weight"] is None:
        metric_cols[3].metric("POS_WEIGHT", "n/a")
    else:
        metric_cols[3].metric("POS_WEIGHT", f"{expected['pos_weight']:.4f}")

    st.subheader("Class Balance")
    class_rows = []
    for split_row in expected["rows"]:
        for class_name in CLASS_NAMES:
            class_rows.append(
                {
                    "split": split_row["split"],
                    "class": class_name,
                    "count": split_row.get(class_name, 0),
                }
            )
    st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
    plot_class_balance(expected["rows"])

    if expected["missing"]:
        with st.expander("Missing required folders", expanded=True):
            st.code("\n".join(expected["missing"]), language="text")

    if expected.get("layout") == "flat":
        st.subheader("Source Class Folders")
        flat = inspect_flat_dataset(str(DATA_DIR))
        st.dataframe(pd.DataFrame(flat["rows"]), use_container_width=True, hide_index=True)

    fallback_roots = alternate_dataset_roots()
    for root in fallback_roots:
        flat = inspect_flat_dataset(str(root))
        if flat["has_classes"]:
            st.subheader(f"Detected Alternate Dataset: {rel_path(root)}")
            st.info(
                "This folder has class folders but not train/val/test splits. "
                "It is shown for inspection; the repo training/evaluation loaders still require the split layout."
            )
            st.dataframe(pd.DataFrame(flat["rows"]), use_container_width=True, hide_index=True)

    st.subheader("Sample Preview")
    try:
        image, label, source = load_training_loader_preview(batch_size=int(getattr(config, "BATCH_SIZE", 4)))
        if image is None:
            st.warning("No sample image could be loaded.")
        else:
            st.image(image, caption=f"Label: {label} ({source})", width=360)
            if source != "training loader":
                st.warning("The image above is a fallback preview from detected dataset files.")
    except Exception as exc:
        st.error(f"Could not load sample batch preview: {exc}")


def render_training() -> None:
    st.title("Training")
    expected = inspect_expected_dataset(str(DATA_DIR))

    if torch is None:
        st.error("PyTorch is not installed. Run `pip install -r requirements.txt` before training.")

    if not expected["complete"]:
        st.warning(
            "Training is disabled until the configured dataset has either train/val/test class folders "
            "or flat NORMAL and PNEUMONIA class directories."
        )

    with st.form("training_controls"):
        cols = st.columns(3)
        model_type = cols[0].selectbox("Model type", MODEL_TYPES, index=0)
        num_epochs = cols[1].number_input(
            "Epochs", min_value=1, max_value=500, value=int(getattr(config, "NUM_EPOCHS", 20)), step=1
        )
        batch_size = cols[2].number_input(
            "Batch size", min_value=1, max_value=256, value=int(getattr(config, "BATCH_SIZE", 32)), step=1
        )

        cols = st.columns(3)
        learning_rate = cols[0].number_input(
            "Learning rate",
            min_value=1e-8,
            max_value=1.0,
            value=float(getattr(config, "LEARNING_RATE", 3e-4)),
            step=1e-5,
            format="%.8f",
        )
        use_lung_segmentation = cols[1].checkbox(
            "Use lung segmentation", value=bool(getattr(config, "USE_LUNG_SEGMENTATION", False))
        )
        pretrained = cols[2].checkbox("Use pretrained weights", value=True)

        with st.expander("Advanced flags", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            use_amp = c1.checkbox("AMP", value=bool(getattr(config, "USE_AMP", False)))
            use_cosine = c2.checkbox("Cosine annealing", value=bool(getattr(config, "USE_COSINE_ANNEALING", True)))
            use_warmup = c3.checkbox("Warmup", value=bool(getattr(config, "USE_WARMUP", True)))
            use_gradient_clipping = c4.checkbox(
                "Gradient clipping", value=bool(getattr(config, "USE_GRADIENT_CLIPPING", True))
            )

            c1, c2, c3, c4 = st.columns(4)
            use_focal_loss = c1.checkbox("Focal loss", value=bool(getattr(config, "USE_FOCAL_LOSS", False)))
            use_mixup = c2.checkbox("Mixup", value=bool(getattr(config, "USE_MIXUP", False)))
            use_cutmix = c3.checkbox("CutMix", value=bool(getattr(config, "USE_CUTMIX", False)))
            use_advanced_aug = c4.checkbox(
                "Advanced augmentation", value=bool(getattr(config, "USE_ADVANCED_AUG", False))
            )

            c1, c2 = st.columns(2)
            gradient_clip_value = c1.number_input(
                "Gradient clip value",
                min_value=0.01,
                max_value=100.0,
                value=float(getattr(config, "GRADIENT_CLIP_VALUE", 1.0)),
                step=0.1,
            )
            dataloader_workers = c2.number_input(
                "DataLoader workers",
                min_value=0,
                max_value=16,
                value=0,
                step=1,
                help="0 is the most reliable setting for Streamlit on Windows.",
            )

        submitted = st.form_submit_button("Start training", disabled=not expected["complete"] or torch is None)

    if not submitted:
        return
    if torch is None:
        st.stop()

    save_dir = MODELS_DIR / f"{model_type}_streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    status = st.empty()
    logs = st.empty()
    status.info("Training started. This runs in-process and may take a while.")

    updates = {
        "USE_AMP": bool(use_amp),
        "USE_COSINE_ANNEALING": bool(use_cosine),
        "USE_WARMUP": bool(use_warmup),
        "USE_GRADIENT_CLIPPING": bool(use_gradient_clipping),
        "GRADIENT_CLIP_VALUE": float(gradient_clip_value),
        "USE_FOCAL_LOSS": bool(use_focal_loss),
        "USE_MIXUP": bool(use_mixup),
        "USE_CUTMIX": bool(use_cutmix),
        "USE_ADVANCED_AUG": bool(use_advanced_aug),
        "USE_LUNG_SEGMENTATION": bool(use_lung_segmentation),
        "NUM_EPOCHS": int(num_epochs),
        "BATCH_SIZE": int(batch_size),
        "LEARNING_RATE": float(learning_rate),
        "NUM_WORKERS": int(dataloader_workers),
        "PERSISTENT_WORKERS": bool(dataloader_workers > 0 and getattr(config, "PERSISTENT_WORKERS", False)),
    }

    try:
        from train import train_model

        with temporary_config(**updates):
            result, log_text = run_with_live_logs(
                train_model,
                model_type=model_type,
                use_lung_mask=use_lung_segmentation,
                pretrained=pretrained,
                num_epochs=int(num_epochs),
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                device="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
                save_dir=str(save_dir),
                log_placeholder=logs,
            )
        status.success("Training completed.")
        st.subheader("Training Results")
        st.metric("Saved model directory", result.get("save_dir", str(save_dir)))
        c1, c2 = st.columns(2)
        c1.metric("Best validation loss", f"{result.get('best_val_loss', 0):.4f}")
        c2.metric("Best validation AUC", f"{result.get('best_val_auc', 0):.4f}")
        st.session_state["last_training_log"] = log_text
        st.cache_data.clear()
    except Exception as exc:
        status.error(f"Training failed: {exc}")
        st.exception(exc)


def render_evaluation() -> None:
    st.title("Evaluation")
    checkpoint_path, inferred_type = checkpoint_selector("Select checkpoint from models/", "eval")
    model_type = st.selectbox(
        "Model type",
        MODEL_TYPES,
        index=MODEL_TYPES.index(inferred_type) if inferred_type in MODEL_TYPES else 0,
        key="eval_model_type",
    )
    split = st.selectbox("Dataset split", ["test", "validation"], index=0)

    expected = inspect_expected_dataset(str(DATA_DIR))
    if not expected["complete"]:
        st.warning("Evaluation requires a valid dataset under config.DATA_DIR.")
    if torch is None:
        st.error("PyTorch is not installed. Run `pip install -r requirements.txt` before evaluation.")

    can_run = checkpoint_path is not None and Path(checkpoint_path).exists() and expected["complete"] and torch is not None
    if checkpoint_path and not Path(checkpoint_path).exists():
        st.error(f"Checkpoint does not exist: {checkpoint_path}")

    if st.button("Run evaluation", disabled=not can_run):
        logs = st.empty()
        try:
            from evaluate import evaluate_trained_model

            result, log_text = run_with_live_logs(
                evaluate_trained_model,
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                use_test_set=(split == "test"),
                device="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
                save_dir=str(EVALUATION_DIR),
                log_placeholder=logs,
            )
            summary_path = write_streamlit_evaluation_summary(result, log_text)
            st.success(f"Evaluation complete. Summary saved to {rel_path(summary_path)}")
            render_metric_cards(result.get("metrics", {}))

            report_text = extract_report_block(log_text)
            st.subheader("Classification Report Summary")
            st.text_area("Report", value=report_text, height=260)

            cols = st.columns(2)
            cm_path = EVALUATION_DIR / f"{model_type}_confusion_matrix.png"
            roc_path = EVALUATION_DIR / f"{model_type}_roc_curve.png"
            if cm_path.exists():
                cols[0].image(str(cm_path), caption="Confusion matrix", use_container_width=True)
            if roc_path.exists():
                cols[1].image(str(roc_path), caption="ROC curve", use_container_width=True)
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            st.exception(exc)


def render_explainability() -> None:
    st.title("Explainability")
    if torch is None:
        st.error("PyTorch is not installed. Run `pip install -r requirements.txt` before explainability runs.")

    checkpoint_path, inferred_type = checkpoint_selector("Select model checkpoint", "explain")
    model_type = st.selectbox(
        "Model type",
        MODEL_TYPES,
        index=MODEL_TYPES.index(inferred_type) if inferred_type in MODEL_TYPES else 0,
        key="explain_model_type",
    )

    tabs = st.tabs(["Single image", "Batch explainability", "Compare two images"])

    with tabs[0]:
        st.subheader("Single Image Prediction")
        uploaded = st.file_uploader("Upload an X-ray image", type=sorted([ext.strip(".") for ext in IMAGE_EXTENSIONS]))
        default_path = default_image_for_class("PNEUMONIA")
        image_path_text = st.text_input("Or choose an image path", value=default_path)
        output_dir = st.text_input("Output directory", value=str(RESULTS_DIR / "explanations"))

        image_path = save_uploaded_image(uploaded) if uploaded is not None else Path(image_path_text) if image_path_text else None
        if image_path and Path(image_path).exists():
            st.image(str(image_path), caption=rel_path(Path(image_path)), width=360)

        can_run = checkpoint_path and Path(checkpoint_path).exists() and image_path and Path(image_path).exists() and torch is not None
        if st.button("Run single-image explanation", disabled=not can_run):
            logs = st.empty()
            start_ts = time.time()
            try:
                from explain_prediction import explain_single_image

                prediction = predict_single_image(str(image_path), str(checkpoint_path), model_type)
                result_cols = st.columns(3)
                result_cols[0].metric("Predicted class", prediction["predicted_class"])
                result_cols[1].metric("Pneumonia probability", f"{prediction['pneumonia_probability']:.4f}")
                result_cols[2].metric("Confidence", f"{prediction['confidence']:.4f}")

                _, log_text = run_with_live_logs(
                    explain_single_image,
                    image_path=str(image_path),
                    model_path=str(checkpoint_path),
                    model_type=model_type,
                    output_dir=output_dir,
                    patient_id=Path(str(image_path)).stem,
                    log_placeholder=logs,
                )
                artifacts = recent_artifacts(Path(output_dir), since_ts=start_ts)
                st.success(f"Explanation complete. Outputs saved to {output_dir}")
                if artifacts:
                    st.dataframe(
                        pd.DataFrame(
                            [{"path": rel_path(p), "modified": format_mtime(p), "bytes": p.stat().st_size} for p in artifacts]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    display_image_grid(artifacts)
                with st.expander("Explainability log"):
                    st.code(log_text, language="text")
            except Exception as exc:
                st.error(f"Single-image explanation failed: {exc}")
                st.exception(exc)

    with tabs[1]:
        st.subheader("Batch Explainability")
        default_folder = best_available_image_folder()
        folder = st.text_input("Folder containing images", value=str(default_folder) if default_folder else "")
        max_samples = st.number_input("Max samples", min_value=1, max_value=500, value=10, step=1)
        output_dir = st.text_input("Batch output directory", value=str(RESULTS_DIR / "explanations" / "batch"))
        can_run = checkpoint_path and Path(checkpoint_path).exists() and folder and Path(folder).is_dir() and torch is not None
        if st.button("Run batch explainability", disabled=not can_run):
            logs = st.empty()
            start_ts = time.time()
            try:
                from explain_prediction import explain_batch

                _, log_text = run_with_live_logs(
                    explain_batch,
                    data_dir=folder,
                    model_path=str(checkpoint_path),
                    model_type=model_type,
                    output_dir=output_dir,
                    max_samples=int(max_samples),
                    log_placeholder=logs,
                )
                artifacts = recent_artifacts(Path(output_dir), since_ts=start_ts)
                st.success(f"Batch explainability complete. Outputs saved to {output_dir}")
                st.metric("Generated files", len(artifacts))
                if artifacts:
                    st.dataframe(
                        pd.DataFrame([{"path": rel_path(p), "modified": format_mtime(p)} for p in artifacts]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    display_image_grid(artifacts)
                with st.expander("Batch log"):
                    st.code(log_text, language="text")
            except Exception as exc:
                st.error(f"Batch explainability failed: {exc}")
                st.exception(exc)

    with tabs[2]:
        st.subheader("Compare Two Images")
        normal_default = default_image_for_class("NORMAL")
        pneumonia_default = default_image_for_class("PNEUMONIA")
        image_a = st.text_input("Image A path", value=normal_default)
        image_b = st.text_input("Image B path", value=pneumonia_default)
        output_dir = st.text_input("Comparison output directory", value=str(RESULTS_DIR / "comparisons"))

        cols = st.columns(2)
        if image_a and Path(image_a).exists():
            cols[0].image(image_a, caption="Image A", use_container_width=True)
        if image_b and Path(image_b).exists():
            cols[1].image(image_b, caption="Image B", use_container_width=True)

        can_run = (
            checkpoint_path
            and Path(checkpoint_path).exists()
            and image_a
            and Path(image_a).exists()
            and image_b
            and Path(image_b).exists()
            and torch is not None
        )
        if st.button("Compare predictions", disabled=not can_run):
            logs = st.empty()
            start_ts = time.time()
            try:
                from explain_prediction import compare_predictions

                _, log_text = run_with_live_logs(
                    compare_predictions,
                    image_path1=image_a,
                    image_path2=image_b,
                    model_path=str(checkpoint_path),
                    model_type=model_type,
                    output_dir=output_dir,
                    log_placeholder=logs,
                )
                artifacts = recent_artifacts(Path(output_dir), since_ts=start_ts)
                st.success(f"Comparison complete. Outputs saved to {output_dir}")
                if artifacts:
                    st.dataframe(
                        pd.DataFrame([{"path": rel_path(p), "modified": format_mtime(p)} for p in artifacts]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    display_image_grid(artifacts)
                with st.expander("Comparison log"):
                    st.code(log_text, language="text")
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
                st.exception(exc)


def render_results() -> None:
    st.title("Results")

    st.subheader("Latest Evaluation Summary")
    latest = latest_evaluation_json()
    if latest is None:
        st.warning("No saved evaluation JSON files found in results/evaluation.")
    else:
        data = read_json(latest)
        st.caption(f"Latest file: {rel_path(latest)}")
        if data and isinstance(data.get("metrics"), dict):
            render_metric_cards(data["metrics"])
        if data:
            st.dataframe(json_to_dataframe(data), use_container_width=True, hide_index=True)

    st.subheader("All Evaluation JSON Files")
    eval_files = sorted(EVALUATION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if EVALUATION_DIR.exists() else []
    if eval_files:
        st.dataframe(
            pd.DataFrame(
                [{"file": rel_path(p), "modified": format_mtime(p), "bytes": p.stat().st_size} for p in eval_files]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Grad-CAM Metrics")
    gradcam_metrics_path = GRADCAM_DIR / "gradcam_metrics.json"
    gradcam_metrics = read_json(gradcam_metrics_path) if gradcam_metrics_path.exists() else None
    if gradcam_metrics:
        st.dataframe(json_to_dataframe(gradcam_metrics), use_container_width=True, hide_index=True)
    else:
        st.info("No Grad-CAM metrics file found at results/gradcam/gradcam_metrics.json.")

    st.subheader("Training Plots")
    plot_files = sorted(TRAINING_PLOTS_DIR.glob("*.png")) if TRAINING_PLOTS_DIR.exists() else []
    if plot_files:
        display_image_grid(plot_files, max_images=12)
    else:
        st.info("No training plots found in results/training_plots.")
    if st.button("Regenerate latest training plots with visualize_training.py"):
        logs = st.empty()
        try:
            from visualize_training import visualize_latest_training

            _, log_text = run_with_live_logs(
                visualize_latest_training,
                models_dir=str(MODELS_DIR),
                log_placeholder=logs,
            )
            st.success("Training plots regenerated in results/training_plots.")
            with st.expander("visualize_training.py log"):
                st.code(log_text, language="text")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Could not regenerate training plots: {exc}")

    st.subheader("Evaluation Visualizations")
    eval_images = sorted(EVALUATION_DIR.glob("*.png")) if EVALUATION_DIR.exists() else []
    if eval_images:
        display_image_grid(eval_images, max_images=8)

    st.subheader("Comprehensive Output Summary")
    summary_rows = [
        {
            "area": "Evaluation",
            "location": rel_path(EVALUATION_DIR),
            "files": len(recent_artifacts(EVALUATION_DIR)),
            "status": "available" if EVALUATION_DIR.exists() else "missing",
        },
        {
            "area": "Grad-CAM",
            "location": rel_path(GRADCAM_DIR),
            "files": len(recent_artifacts(GRADCAM_DIR)),
            "status": "available" if GRADCAM_DIR.exists() else "missing",
        },
        {
            "area": "Training plots",
            "location": rel_path(TRAINING_PLOTS_DIR),
            "files": len(recent_artifacts(TRAINING_PLOTS_DIR)),
            "status": "available" if TRAINING_PLOTS_DIR.exists() else "missing",
        },
        {
            "area": "Explainability demo",
            "location": rel_path(RESULTS_DIR / "explainability_demo"),
            "files": len(recent_artifacts(RESULTS_DIR / "explainability_demo")),
            "status": "available" if (RESULTS_DIR / "explainability_demo").exists() else "missing",
        },
        {
            "area": "Ablation",
            "location": rel_path(RESULTS_DIR / "ablation"),
            "files": len(recent_artifacts(RESULTS_DIR / "ablation")),
            "status": "available" if (RESULTS_DIR / "ablation").exists() else "missing",
        },
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader("Repository Result Scripts")
    script_cols = st.columns(2)
    if script_cols[0].button("Run show_results.py text summary"):
        try:
            output = run_repo_script("show_results.py")
            st.text_area("show_results.py output", value=output, height=260)
        except Exception as exc:
            st.error(f"Could not run show_results.py: {exc}")

    if script_cols[1].button("Run show_all_results.py text summary"):
        try:
            if display_all_results is not None:
                logs = st.empty()
                _, output = run_with_live_logs(display_all_results, log_placeholder=logs)
            else:
                output = run_repo_script("show_all_results.py")
            st.text_area("show_all_results.py output", value=output, height=360)
        except Exception as exc:
            st.error(f"Could not run show_all_results.py summary: {exc}")


def render_model_manager() -> None:
    st.title("Model Manager")
    folders = list_model_folders(str(MODELS_DIR))

    if st.button("Refresh model list"):
        st.cache_data.clear()
        st.rerun()

    if not folders:
        st.warning(f"No model folders found in {MODELS_DIR}.")
        return

    rows = []
    for folder in folders:
        rows.append(
            {
                "folder": rel_path(Path(folder["folder"])),
                "model_type": folder["model_type"],
                "checkpoint_count": folder["checkpoint_count"],
                "best_model": rel_path(Path(folder["best_model"])) if folder["best_model"] else "",
                "final_model": rel_path(Path(folder["final_model"])) if folder["final_model"] else "",
                "training_history": rel_path(Path(folder["history"])) if folder["history"] else "",
                "modified": folder["modified"],
            }
        )

    st.subheader("Available Model Folders")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    checkpoints = list_checkpoints()
    st.subheader("Checkpoint Selector")
    if checkpoints:
        latest = checkpoints[0]
        st.success(f"Latest checkpoint: {rel_path(Path(latest['path']))}")
        options = [item["label"] for item in checkpoints]
        selected_label = st.selectbox("Select checkpoint", options, key="manager_checkpoint")
        selected = checkpoints[options.index(selected_label)]
        selected_path = Path(selected["path"])
        st.write(f"Path: `{selected_path}`")
        st.write(f"Inferred model type: `{selected['model_type']}`")
        st.write(f"Modified: `{format_mtime(selected_path)}`")

        if st.button("Load checkpoint metadata"):
            if torch is None:
                st.error("PyTorch is not available, so checkpoint metadata cannot be loaded.")
            else:
                try:
                    checkpoint = torch.load(str(selected_path), map_location="cpu", weights_only=False)
                    if isinstance(checkpoint, dict):
                        metadata = {
                            key: value
                            for key, value in checkpoint.items()
                            if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
                        }
                        st.dataframe(json_to_dataframe(metadata), use_container_width=True, hide_index=True)
                    else:
                        st.info(f"Checkpoint object type: {type(checkpoint).__name__}")
                except Exception as exc:
                    st.error(f"Could not load checkpoint metadata: {exc}")
    else:
        st.warning("No checkpoint files were found. The folders above currently contain training histories only.")

    st.subheader("Training History Metadata")
    folder_labels = [rel_path(Path(folder["folder"])) for folder in folders]
    selected_folder_label = st.selectbox("Select model folder", folder_labels)
    selected_folder = folders[folder_labels.index(selected_folder_label)]
    if selected_folder["history"]:
        history = read_json(Path(selected_folder["history"]))
        if history:
            st.caption(rel_path(Path(selected_folder["history"])))
            history_lengths = {
                key: len(value) if isinstance(value, list) else "n/a" for key, value in history.items()
            }
            st.dataframe(json_to_dataframe(history_lengths), use_container_width=True, hide_index=True)
            if all(k in history for k in ["train_loss", "val_loss"]) and history["train_loss"]:
                fig, ax = plt.subplots(figsize=(8, 4))
                epochs = range(1, len(history["train_loss"]) + 1)
                ax.plot(epochs, history["train_loss"], label="Train loss")
                ax.plot(epochs, history["val_loss"], label="Validation loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training History")
                ax.legend()
                ax.grid(alpha=0.25)
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("No training_history.json file found for the selected folder.")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Menu",
        ["Home", "Dataset", "Training", "Evaluation", "Explainability", "Results", "Model Manager"],
    )

    st.sidebar.divider()
    st.sidebar.caption("Runtime")
    if torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.write(f"Device: `{device}`")
        if torch.cuda.is_available():
            st.sidebar.write(f"GPU: `{torch.cuda.get_device_name(0)}`")
    else:
        st.sidebar.write("Device: `torch unavailable`")

    st.sidebar.caption(f"Project: `{PROJECT_ROOT.name}`")

    if page == "Home":
        render_home()
    elif page == "Dataset":
        render_dataset()
    elif page == "Training":
        render_training()
    elif page == "Evaluation":
        render_evaluation()
    elif page == "Explainability":
        render_explainability()
    elif page == "Results":
        render_results()
    elif page == "Model Manager":
        render_model_manager()


if __name__ == "__main__":
    main()
