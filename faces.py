"""Распознавание лиц: детекция (SCRFD) + эмбеддинги (ArcFace), всё локально.

Модели скачиваются один раз с релиза insightface и хранятся в ./models/.
Известные люди — в ./faces.db (SQLite). Работает полностью офлайн.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger("faces")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "faces.db"

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_FILE = MODEL_DIR / "det_10g.onnx"
REC_FILE = MODEL_DIR / "w600k_r50.onnx"

DET_SIZE = 640
DET_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MIN_FACE_PX = 40
MAX_FACES = 5
MAX_EMB_PER_PERSON = 8
SIM_THRESHOLD = 0.45
EMB_DIM = 512

# Стандартный шаблон выравнивания ArcFace 112x112.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_STRIDES = (8, 16, 32)

_engine_lock = threading.Lock()
_sess_det = None
_sess_rec = None
# До какого момента (monotonic) не пытаться трогать модели после жёсткой ошибки.
_unavailable_until = 0.0
_UNAVAILABLE_COOLDOWN = 3600.0


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------

def models_present() -> bool:
    return DET_FILE.is_file() and REC_FILE.is_file()


def _download_with_progress(url: str, dest: Path, chunk: int = 1024 * 1024) -> None:
    last_pct = -1

    def hook(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct
        if total and total > 0:
            pct = int(blocks * block_size * 100 / total)
            if pct >= last_pct + 10:
                last_pct = pct
                log.info("Скачивание моделей лиц: %s%%", pct)

    urllib.request.urlretrieve(url, str(dest), reporthook=hook)


def ensure_models() -> tuple[Path, Path]:
    """Скачивает и распаковывает модели при первом использовании."""
    global _unavailable_until
    if models_present():
        return DET_FILE, REC_FILE
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODEL_DIR / "buffalo_l.zip"
    try:
        log.info("Скачиваю модели распознавания лиц (~340 МБ, один раз)...")
        _download_with_progress(PACK_URL, zip_path)
        log.info("Распаковываю модели лиц...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base in (DET_FILE.name, REC_FILE.name):
                    dest = MODEL_DIR / base
                    if not dest.exists():
                        with zf.open(name) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
        if not models_present():
            raise RuntimeError("в архиве нет нужных файлов моделей")
        log.info("Модели лиц готовы")
    except Exception:
        _unavailable_until = time.monotonic() + _UNAVAILABLE_COOLDOWN
        raise
    finally:
        try:
            if zip_path.exists():
                zip_path.unlink()
        except OSError:
            pass
    return DET_FILE, REC_FILE


def _get_sessions():
    """Ленивое создание ONNX-сессий (потокобезопасно)."""
    global _sess_det, _sess_rec
    if _sess_det is not None and _sess_rec is not None:
        return _sess_det, _sess_rec
    with _engine_lock:
        if _sess_det is not None and _sess_rec is not None:
            return _sess_det, _sess_rec
        import onnxruntime as ort

        det_path, rec_path = ensure_models()
        opts = ort.SessionOptions()
        try:
            threads = max(1, (os.cpu_count() or 4) // 2)
        except Exception:
            threads = 2
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        log.info("Загружаю модели лиц (CPU, потоков: %s)...", threads)
        t0 = time.monotonic()
        _sess_det = ort.InferenceSession(str(det_path), sess_options=opts, providers=providers)
        _sess_rec = ort.InferenceSession(str(rec_path), sess_options=opts, providers=providers)
        log.info("Модели лиц загружены за %.1f c", time.monotonic() - t0)
        return _sess_det, _sess_rec


async def warmup() -> bool:
    """Прогрев моделей в фоне, чтобы первое фото не ждало скачивания."""
    loop = asyncio.get_running_loop()

    def _load() -> bool:
        try:
            _get_sessions()
            return True
        except Exception:
            log.warning("Модели лиц недоступны, работаю без распознавания", exc_info=True)
            return False

    return await loop.run_in_executor(None, _load)


# ---------------------------------------------------------------------------
# Детекция (SCRFD)
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: list, threshold: float = NMS_THRESHOLD) -> list:
    dets = sorted(dets, key=lambda d: -d[0])
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if _iou(best[1:5], d[1:5]) < threshold]
    return kept


def _decode_level(
    scores: np.ndarray,
    deltas: np.ndarray,
    kpss: np.ndarray,
    stride: int,
) -> list:
    """Декодирует один уровень FPN. Выходы модели — уже вероятности (без sigmoid).

    Якоря — это центры клеток сетки (2 якоря на точку), бокс получается
    сдвигом центра на предсказанные смещения, умноженные на stride.
    """
    gh = gw = DET_SIZE // stride
    ys, xs = np.mgrid[:gh, :gw]
    centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    centers = np.repeat(centers.reshape(-1, 2), 2, axis=0)
    scores = scores.reshape(-1)
    deltas = deltas.reshape(-1, 4)
    kpss = kpss.reshape(-1, 10)
    assert centers.shape[0] == scores.shape[0] == deltas.shape[0] == kpss.shape[0]
    keep = np.where(scores >= DET_THRESHOLD)[0]
    out = []
    for i in keep:
        cx, cy = centers[i]
        x1 = cx - deltas[i, 0] * stride
        y1 = cy - deltas[i, 1] * stride
        x2 = cx + deltas[i, 2] * stride
        y2 = cy + deltas[i, 3] * stride
        kps = np.zeros((5, 2), dtype=np.float32)
        for j in range(5):
            kps[j, 0] = cx + kpss[i, j * 2] * stride
            kps[j, 1] = cy + kpss[i, j * 2 + 1] * stride
        out.append((float(scores[i]), x1, y1, x2, y2, kps))
    return out


def detect_faces_sync(image_bytes: bytes, max_faces: int = MAX_FACES) -> list[dict]:
    """Находит лица на фото. Возвращает список {bbox, kps, score} в пикселях оригинала."""
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    h0, w0 = img.shape[:2]
    scale = DET_SIZE / max(h0, w0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    blob = ((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
    sess_det, _ = _get_sessions()
    try:
        outs = sess_det.run(None, {sess_det.get_inputs()[0].name: blob})
    except Exception:
        log.debug("SCRFD inference не удался", exc_info=True)
        return []
    dets = []
    for lvl, stride in enumerate(_STRIDES):
        dets.extend(_decode_level(outs[lvl], outs[lvl + 3], outs[lvl + 6], stride))
    kept = _nms(dets)
    result = []
    h, w = img.shape[:2]
    for score, x1, y1, x2, y2, kps in kept[:max_faces]:
        x1c, y1c = max(0.0, x1 / scale), max(0.0, y1 / scale)
        x2c, y2c = min(float(w), x2 / scale), min(float(h), y2 / scale)
        if x2c - x1c < MIN_FACE_PX or y2c - y1c < MIN_FACE_PX:
            continue
        result.append(
            {
                "score": float(score),
                "bbox": (x1c, y1c, x2c, y2c),
                "kps": (kps / scale).astype(np.float32),
                "img": img,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Эмбеддинги (ArcFace)
# ---------------------------------------------------------------------------

def align_face(img_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray | None:
    """Выравнивает лицо 112x112 по 5 точкам. None при неудаче."""
    import cv2

    try:
        mat, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    except Exception:
        return None
    if mat is None:
        return None
    try:
        return cv2.warpAffine(img_bgr, mat, (112, 112), borderValue=0.0)
    except Exception:
        return None


def embed_face(aligned_bgr: np.ndarray) -> np.ndarray | None:
    """512-d L2-нормализованный эмбеддинг выровненного лица 112x112."""
    _, sess_rec = _get_sessions()
    try:
        blob = ((aligned_bgr[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        out = sess_rec.run(None, {sess_rec.get_inputs()[0].name: blob})[0]
    except Exception:
        log.debug("ArcFace inference не удался", exc_info=True)
        return None
    vec = np.asarray(out, dtype=np.float32).ravel()
    if vec.shape[0] != EMB_DIM:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return (vec / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


# ---------------------------------------------------------------------------
# База лиц (SQLite)
# ---------------------------------------------------------------------------

MAX_EMB_PER_PERSON = 8


def _connect():
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS persons("
        "id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings("
        "id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,"
        "vec BLOB NOT NULL, source TEXT DEFAULT '', created REAL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id)")
    return conn


def _get_person_id(conn, name: str) -> int | None:
    row = conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
    return int(row[0]) if row else None


def enroll(name: str, embedding: np.ndarray, source: str = "manual") -> int:
    """Сохраняет эмбеддинг человека. Возвращает person_id."""
    name = (name or "").strip()
    if not name:
        raise ValueError("пустое имя")
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    if vec.shape[0] != EMB_DIM:
        raise ValueError("неверная размерность эмбеддинга")
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    now = time.time()
    with _connect() as conn:
        pid = _get_person_id(conn, name)
        if pid is None:
            cur = conn.execute("INSERT INTO persons(name, created) VALUES (?, ?)", (name, now))
            pid = int(cur.lastrowid)
        rows = conn.execute("SELECT vec FROM embeddings WHERE person_id = ?", (pid,)).fetchall()
        for (blob,) in rows:
            old = np.frombuffer(blob, dtype=np.float32)
            if old.shape == (EMB_DIM,) and cosine_similarity(old, vec) > 0.95:
                return pid
        conn.execute(
            "INSERT INTO embeddings(person_id, vec, source, created) VALUES (?, ?, ?, ?)",
            (pid, vec.tobytes(), source, now),
        )
        conn.execute(
            "DELETE FROM embeddings WHERE id NOT IN ("
            "SELECT id FROM embeddings WHERE person_id = ? ORDER BY id DESC LIMIT ?)",
            (pid, MAX_EMB_PER_PERSON),
        )
        conn.commit()
    log.info("Запомнено лицо: %s (%s)", name, source)
    return pid


def identify(embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
    """Лучшее совпадение по всем сохранённым эмбеддингам. (имя|None, score)."""
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    if vec.shape[0] != EMB_DIM:
        return None, 0.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    best_name: str | None = None
    best_score = 0.0
    with _connect() as conn:
        rows = conn.execute(
            "SELECT p.name, e.vec FROM embeddings e JOIN persons p ON p.id = e.person_id"
        ).fetchall()
    for name, blob in rows:
        old = np.frombuffer(blob, dtype=np.float32)
        if old.shape != (EMB_DIM,):
            continue
        score = cosine_similarity(old, vec)
        if score > best_score:
            best_score = score
            best_name = str(name)
    if best_name is not None and best_score >= threshold:
        return best_name, float(best_score)
    return None, float(best_score)


def forget(name: str) -> bool:
    """Удаляет человека и все его эмбеддинги. True если было что удалять."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM persons WHERE name = ?", (name,))
        conn.commit()
        removed = cur.rowcount > 0
    if removed:
        log.info("Забыто лицо: %s", name)
    return removed


def list_persons() -> list[tuple[str, int]]:
    """[(имя, число фото), ...] по алфавиту."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT p.name, COUNT(e.id) FROM persons p "
            "LEFT JOIN embeddings e ON e.person_id = p.id "
            "GROUP BY p.id ORDER BY p.name"
        ).fetchall()
    return [(str(n), int(c)) for n, c in rows]


# ---------------------------------------------------------------------------
# Высокоуровневый анализ одного фото
# ---------------------------------------------------------------------------

async def analyze_photo_async(data: bytes, threshold: float) -> list[dict]:
    """Детекция + эмбеддинги + распознавание. Возвращает список лиц:
    {"name": str|None, "score": float, "det": float, "bbox": tuple, "embedding": np.ndarray|None}."""
    import asyncio

    if time.monotonic() < _unavailable_until:
        return []
    loop = asyncio.get_running_loop()

    def _work():
        faces = []
        for det in detect_faces_sync(data):
            emb = None
            aligned = align_face(det["img"], det["kps"])
            if aligned is not None:
                emb = embed_face(aligned)
            name, score = (None, 0.0)
            if emb is not None:
                name, score = identify(emb, threshold)
            faces.append({"name": name, "score": score, "det": det["score"], "bbox": det["bbox"], "embedding": emb})
        return faces

    try:
        return await loop.run_in_executor(None, _work)
    except Exception:
        log.warning("Анализ фото не удался", exc_info=True)
        return []


def format_people_line(faces: list[dict]) -> str | None:
    """Строка вида 'Люди на фото: Лиска (0.91), неизвестный человек'."""
    if not faces:
        return None
    parts = []
    unknowns = 0
    for f in faces:
        if f.get("name"):
            parts.append(f"{f['name']} ({f.get('score', 0.0):.2f})")
        else:
            unknowns += 1
    if unknowns:
        parts.append("неизвестный человек" if unknowns == 1 else f"неизвестных: {unknowns}")
    if not parts:
        return None
    return "Люди на фото: " + ", ".join(parts)
