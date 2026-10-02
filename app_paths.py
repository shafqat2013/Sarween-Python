"""Writable user data, read-only resources, and a non-destructive v1 migration.

Path lookup and imports never create files. Live entry points explicitly call
initialize_user_data; previews and offline tests can remain read-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


STATE_FILES = (
    "hardware_config.json", "combo_profiles.json", "band_profiles.json",
    "mini_library.json", "mini_token_map.json", "camera_matrix.npy",
    "dist_coeffs.npy", "mini_database.csv",
)
MIGRATION_FILE = "migration-v1.json"


def data_dir() -> Path:
    override = os.environ.get("SARWEEN_DATA_DIR")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError("SARWEEN_DATA_DIR must be an absolute path")
        return path
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Sarween"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Sarween"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Sarween"


def data_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("User-data paths must stay within the data directory")
    return data_dir() / relative


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def resource_path(name: str) -> str:
    return str(resource_root() / name)


def recordings_dir() -> Path:
    path = data_path("Recordings")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")


def _validate_state(path: Path) -> None:
    _regular_file(path)
    if path.name in STATE_FILES and path.suffix == ".json":
        if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
            raise ValueError(f"Expected a JSON object; refusing to replace data: {path}")


def _atomic_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _regular_file(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_bytes(path, contents: bytes, *, backup: bool = True) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        _regular_file(path)
        if backup:
            _atomic_bytes(path.with_suffix(path.suffix + ".bak"), path.read_bytes())
    _atomic_bytes(path, contents)


def atomic_write_json(path, value, *, backup: bool = True) -> None:
    contents = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, contents, backup=backup)


def _legacy_dirs(explicit=None) -> list[Path]:
    if explicit is not None:
        return list(dict.fromkeys(Path(p).expanduser().resolve() for p in explicit))
    # An override is an isolated development/test profile unless import is explicit.
    if os.environ.get("SARWEEN_DATA_DIR"):
        return []
    roots = [resource_root().resolve()]
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    return list(dict.fromkeys(roots))


def migration_plan(legacy_dirs=None, *, destination=None) -> dict:
    root = Path(destination or data_dir()).expanduser().resolve()
    sources = _legacy_dirs(legacy_dirs)
    entries = {}
    for source in sources:
        if source == root or root.is_relative_to(source / "mini_captures"):
            raise ValueError("Migration destination must be separate from legacy data")
        files = [source / name for name in STATE_FILES]
        files += [source / (name + ".bak") for name in STATE_FILES]
        captures = source / "mini_captures"
        if captures.is_symlink():
            raise ValueError(f"Symlinked captures require an explicit manual import: {captures}")
        if captures.is_dir():
            for path in sorted(captures.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Symlinked capture is not imported: {path}")
                if path.is_file():
                    files.append(path)
        for path in files:
            if not path.exists() and not path.is_symlink():
                continue
            relative = path.relative_to(source).as_posix()
            if relative in entries:
                continue
            _validate_state(path)
            entries[relative] = {"path": relative, "source": str(path),
                                 "bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {"schemaVersion": 1, "destination": str(root),
            "legacyRoots": [str(p) for p in sources], "files": list(entries.values())}


@contextmanager
def _migration_lock(root):
    lock = root / ".migration.lock"
    if lock.exists() or lock.is_symlink():
        _regular_file(lock)
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.write(b"\0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _copy_new(entry, target):
    if target.exists() or target.is_symlink():
        _validate_state(target)
        return "preserved-existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output, Path(entry["source"]).open("rb") as source:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(Path(temporary)) != entry["sha256"]:
            raise RuntimeError(f"Source changed during migration: {entry['source']}")
        try:
            # Publish a complete file without ever replacing an existing target.
            os.link(temporary, target)
        except FileExistsError:
            _validate_state(target)
            return "preserved-existing"
        return "copied"
    finally:
        Path(temporary).unlink(missing_ok=True)


def initialize_user_data(legacy_dirs=None, *, destination=None) -> dict:
    root = Path(destination or data_dir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _migration_lock(root):
        for name in STATE_FILES:
            path = root / name
            if path.exists() or path.is_symlink():
                _validate_state(path)
        marker = root / MIGRATION_FILE
        if marker.exists() or marker.is_symlink():
            _regular_file(marker)
            result = json.loads(marker.read_text(encoding="utf-8"))
            if result.get("schemaVersion") != 1 or result.get("complete") is not True:
                raise ValueError(f"Invalid migration record: {marker}")
            return result
        result = migration_plan(legacy_dirs, destination=root)
        for entry in result["files"]:
            target = root / entry["path"]
            for parent in target.parents:
                if parent == root:
                    break
                if parent.is_symlink():
                    raise ValueError(f"Symlinked migration destination: {parent}")
            entry["status"] = _copy_new(entry, target)
            entry["destinationSha256"] = _sha256(target)
        result.update(complete=True, completedAt=datetime.now(timezone.utc).isoformat())
        atomic_write_json(marker, result, backup=False)
        return result


def _migrated_roots() -> list[Path]:
    try:
        data = json.loads(data_path(MIGRATION_FILE).read_text(encoding="utf-8"))
        return [Path(p) for p in data["legacyRoots"]]
    except FileNotFoundError:
        return []


def saved_map_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for root in _migrated_roots():
        candidate = root / path
        if candidate.is_file():
            return str(candidate)
    return resource_path(value)


def capture_asset_path(value: str, db_path) -> str:
    """Resolve old CSV references to copied captures, without rewriting the CSV."""
    path, base = Path(value), Path(db_path).parent
    if base.resolve() == data_dir().resolve():
        for root in _migrated_roots():
            if path.is_absolute() and path.resolve().is_relative_to(root / "mini_captures"):
                return str(base / path.resolve().relative_to(root))
    return str(path if path.is_absolute() else base / path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "migrate"))
    parser.add_argument("--legacy-dir", type=Path, action="append")
    args = parser.parse_args(argv)
    if args.command == "check":
        result = migration_plan(args.legacy_dir)
    else:
        result = initialize_user_data(args.legacy_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
