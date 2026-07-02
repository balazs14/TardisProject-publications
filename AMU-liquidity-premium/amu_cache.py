from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq


CACHE_VERSION = 1
PARQUET_BATCH_ROWS = 100_000


def build_shared_cache_path(base_dir: str | Path, from_date: str, to_date: str) -> Path:
	return Path(base_dir) / f"amu_frames_{from_date}_to_{to_date}_meta.pkl"


def get_cached_frame(cache_path: str | Path, key: str) -> pl.DataFrame | None:
	frame_path = get_cached_frame_parquet_path(cache_path, key)
	if frame_path is None:
		return None

	try:
		return _read_parquet_in_batches(frame_path)
	except Exception:
		return None
	return None


def put_cached_frame(
	cache_path: str | Path,
	key: str,
	frame: pl.DataFrame,
	*,
	from_date: str | None,
	to_date: str | None,
) -> None:
	path = Path(cache_path)
	payload = _read_payload(path)
	payload["version"] = CACHE_VERSION
	payload["from_date"] = from_date
	payload["to_date"] = to_date
	frames = payload.setdefault("frames", {})

	frame_path = _frame_path_for_key(path, key)
	_write_parquet_in_batches(frame, frame_path)
	frames[key] = _frame_entry(frame_path, frame.height, frame.width)

	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("wb") as handle:
		pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
	tmp_path.replace(path)


def get_cached_frame_parquet_path(cache_path: str | Path, key: str) -> Path | None:
	path = Path(cache_path)
	if not path.exists():
		return None
	payload = _read_payload(path)
	frames = payload.get("frames", {})
	entry = frames.get(key)
	if not isinstance(entry, dict):
		# Backward compatibility for older cache files that embedded DataFrames in metadata.
		if isinstance(entry, pl.DataFrame):
			put_cached_frame(
				cache_path,
				key,
				entry,
				from_date=payload.get("from_date"),
				to_date=payload.get("to_date"),
			)
			return _frame_path_for_key(path, key)
		return None

	rel_path = entry.get("parquet_relpath")
	if not isinstance(rel_path, str):
		return None

	frame_path = path.parent / rel_path
	if not frame_path.exists():
		return None
	return frame_path


def iter_cached_frame_batches(cache_path: str | Path, key: str, batch_rows: int = PARQUET_BATCH_ROWS):
	frame_path = get_cached_frame_parquet_path(cache_path, key)
	if frame_path is None:
		return
	parquet = pq.ParquetFile(str(frame_path))
	for batch in parquet.iter_batches(batch_size=batch_rows):
		yield pl.from_arrow(batch)


def put_cached_frame_parquet_path(
	cache_path: str | Path,
	key: str,
	*,
	frame_path: str | Path,
	n_rows: int,
	n_cols: int,
	from_date: str | None,
	to_date: str | None,
) -> None:
	path = Path(cache_path)
	payload = _read_payload(path)
	payload["version"] = CACHE_VERSION
	payload["from_date"] = from_date
	payload["to_date"] = to_date
	frames = payload.setdefault("frames", {})

	resolved = Path(frame_path).resolve()
	try:
		rel_path = resolved.relative_to(path.parent.resolve())
		stored_path = str(rel_path)
	except ValueError:
		stored_path = str(resolved)

	frames[key] = {
		"parquet_relpath": stored_path,
		"n_rows": int(n_rows),
		"n_cols": int(n_cols),
	}

	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("wb") as handle:
		pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
	tmp_path.replace(path)


def _read_payload(path: Path) -> dict[str, Any]:
	if not path.exists():
		return _empty_payload()
	try:
		with path.open("rb") as handle:
			payload = pickle.load(handle)
		if isinstance(payload, dict) and isinstance(payload.get("frames", {}), dict):
			return payload
	except Exception:
		return _empty_payload()
	return _empty_payload()


def _empty_payload() -> dict[str, Any]:
	return {"version": CACHE_VERSION, "frames": {}}


def _frame_path_for_key(meta_path: Path, key: str) -> Path:
	safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "frame"
	base_name = meta_path.name.removesuffix("_meta.pkl")
	return meta_path.parent / f"{base_name}__{safe_key}.parquet"


def _frame_entry(frame_path: Path, n_rows: int, n_cols: int) -> dict[str, object]:
	return {
		"parquet_relpath": frame_path.name,
		"n_rows": int(n_rows),
		"n_cols": int(n_cols),
	}


def _write_parquet_in_batches(frame: pl.DataFrame, target_path: Path, batch_rows: int = PARQUET_BATCH_ROWS) -> None:
	target_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
	writer: pq.ParquetWriter | None = None
	try:
		for start in range(0, frame.height, batch_rows):
			chunk = frame.slice(start, batch_rows)
			table = chunk.to_arrow()
			if writer is None:
				writer = pq.ParquetWriter(str(tmp_path), table.schema, compression="zstd")
			writer.write_table(table)
		if writer is None:
			# Keep behavior deterministic even for empty frames.
			frame.write_parquet(tmp_path)
		else:
			writer.close()
			writer = None
			tmp_path.replace(target_path)
			return
		tmp_path.replace(target_path)
	finally:
		if writer is not None:
			writer.close()


def _read_parquet_in_batches(source_path: Path, batch_rows: int = PARQUET_BATCH_ROWS) -> pl.DataFrame:
	parquet = pq.ParquetFile(str(source_path))
	chunks: list[pl.DataFrame] = []
	for batch in parquet.iter_batches(batch_size=batch_rows):
		chunks.append(pl.from_arrow(batch))
	if not chunks:
		return pl.DataFrame()
	if len(chunks) == 1:
		return chunks[0]
	return pl.concat(chunks, how="vertical_relaxed", rechunk=False)