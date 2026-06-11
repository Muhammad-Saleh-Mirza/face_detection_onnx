import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

MODELS_URL = "https://ai-public-videos.s3.us-east-2.amazonaws.com/Inferenced+Videos/_weights/models.zip"


def _extract_strip_toplevel(zip_path: Path, dest_dir: Path) -> None:
    """
    Extract a zip archive into dest_dir, automatically stripping a single
    top-level directory if the archive contains one.

    Example:
        Archive layout:  models/buffalo_l/det_10g.onnx
        dest_dir:        /app/models
        Result:          /app/models/buffalo_l/det_10g.onnx   (no duplication)
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        # Collect the set of unique top-level names inside the archive
        top_level: set[str] = set()
        for m in members:
            parts = Path(m.filename).parts
            if parts:
                top_level.add(parts[0])

        # If every entry shares exactly one top-level directory, strip it
        strip_prefix = ""
        if len(top_level) == 1:
            strip_prefix = list(top_level)[0] + "/"
            log.info("Detected top-level archive prefix '%s' — stripping during extraction", strip_prefix)

        for member in members:
            rel = member.filename

            if strip_prefix and rel.startswith(strip_prefix):
                rel = rel[len(strip_prefix):]

            # Skip directory entries and the stripped root itself
            if not rel or rel.endswith("/"):
                continue

            out_path = dest_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(out_path, "wb") as dst:
                while chunk := src.read(1024 * 256):
                    dst.write(chunk)

            log.debug("Extracted: %s", out_path)


def download_and_extract_models(dest_dir: str = "models") -> None:
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    zip_target = dest_path / "models.zip"

    last_pct = [-1]

    def _progress(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(int(count * block_size * 100 / total_size), 100)
        if pct - last_pct[0] >= 5:
            last_pct[0] = pct
            filled  = pct // 5
            bar     = "█" * filled + "░" * (20 - filled)
            done_mb = count * block_size / 1024 / 1024
            tot_mb  = total_size / 1024 / 1024
            log.info("[%s] %d%%  %.1f / %.1f MB", bar, pct, done_mb, tot_mb)

    log.info("Downloading model weights from %s → %s", MODELS_URL, zip_target)
    try:
        urllib.request.urlretrieve(MODELS_URL, str(zip_target), reporthook=_progress)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise RuntimeError(
                f"S3 download failed (403 Forbidden). The bucket may be private.\n"
                f"Fix: manually copy the buffalo_l folder to {dest_path / 'buffalo_l'} "
                f"or update MODELS_URL in services/model_loader.py."
            ) from e
        raise
    log.info("Download complete. Extracting...")

    _extract_strip_toplevel(zip_target, dest_path)

    zip_target.unlink()
    log.info("Model weights ready at %s", dest_path)


def ensure_models(dest_dir: str = "models") -> tuple[str, str]:
    """
    Ensures both ONNX models are present.
    Returns (det_model_path, rec_model_path).
    """
    dest_path = Path(dest_dir).resolve()
    det_onnx = dest_path / "buffalo_l" / "det_10g.onnx"
    rec_onnx = dest_path / "buffalo_l" / "w600k_r50.onnx"

    if not det_onnx.exists() or not rec_onnx.exists():
        duplicate = dest_path / dest_path.name / "buffalo_l"
        if duplicate.exists():
            log.warning(
                "Found duplicate model folder at %s — this is from a previous bad extraction. "
                "Remove it manually or re-run after cleaning %s.",
                duplicate.parent, dest_path,
            )

        missing = []
        if not det_onnx.exists():
            missing.append("det_10g.onnx")
        if not rec_onnx.exists():
            missing.append("w600k_r50.onnx")
        log.warning("Missing models %s — downloading...", missing)
        download_and_extract_models(str(dest_path))
    else:
        log.info("Models ready: %s, %s", det_onnx, rec_onnx)

    return str(det_onnx), str(rec_onnx)
