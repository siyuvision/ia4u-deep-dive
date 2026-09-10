"""Windows execution adapter; upstream model, thresholds and NMS are unchanged.

OpenSlide handles cannot be pickled by Windows spawn. Read image batches in
the parent process; use threads for CPU postprocessing instead of processes.
This is an adapted local pipeline benchmark, not the author's Linux benchmark.

Windows also refuses to delete SQLite files while a connection is open
(WinError 32).  Every SQLiteStore here lives in this process (threads), so on a
delete failure we close any connection bound to the target file and retry.
"""
import gc
import os
import shutil as _real_shutil
import sqlite3
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'vendor/KongNet_Inference_Main')
)


def _close_connections_on(path):
    target = os.path.abspath(str(path)).lower()
    for obj in gc.get_objects():
        if type(obj) is sqlite3.Connection:
            try:
                rows = obj.execute('PRAGMA database_list').fetchall()
            except Exception:
                continue
            for row in rows:
                try:
                    db = os.path.abspath(str(row[2])).lower() if row[2] else ''
                except Exception:
                    db = ''
                if db and db.startswith(target):
                    try:
                        obj.close()
                    except Exception:
                        pass
                    break


def _safe_remove(path):
    """Remove one file, closing any in-process SQLite connection to it first.

    Only used at the official pipeline's temp-db removal points.  Never
    monkey-patch os.remove globally: scipy and other Cython extensions pass
    bytes paths and retry logic corrupts them (WinError 123).

    Non-fatal: if removal fails after retries, log a warning and continue.
    The temp file lives in the cache directory which is cleaned up separately.
    Losing a result because a temp-file cleanup failed is unacceptable.
    """
    for attempt in range(40):
        try:
            os.remove(path)
            return
        except PermissionError:
            _close_connections_on(path)
            gc.collect()
            time.sleep(0.5)
    print(f'WARNING: could not remove temp file after 40 retries: {path}', flush=True)


def _safe_rmtree(path):
    """Best-effort rmtree used only for our own temp cache at exit.

    Non-fatal: results are already saved; losing them because a cache
    temp file is still locked is unacceptable.
    """
    for attempt in range(40):
        try:
            _real_shutil.rmtree(path)
            return
        except PermissionError:
            _close_connections_on(path)
            gc.collect()
            time.sleep(0.5)
    print(f'WARNING: could not remove cache dir after 40 retries: {path}', flush=True)


class _OSNamespace:
    """Module-level os replacement for data_utils only.

    The official slide_nms calls os.remove directly on its temp SQLite files.
    Windows keeps those files locked while a connection is open, so only that
    one call is redirected through _safe_remove; every other os attribute is
    delegated to the real module unchanged.  Binding this only inside
    data_utils leaves the shared os module untouched (bytes-path callers such
    as scipy's Cython extensions are unaffected).
    """

    def remove(self, path):
        _safe_remove(path)

    def __getattr__(self, name):
        return getattr(os, name)


class _ShutilNamespace:
    """Module-level shutil replacement for cache cleanup call sites."""

    def rmtree(self, path):
        _safe_rmtree(path)

    def __getattr__(self, name):
        return getattr(_real_shutil, name)

if __name__ == '__main__':
    from inference import wsi_inference_base as base
    from inference import data_utils
    from inference import base_inference_interface as interface

    original_loader = base.DataLoader

    def windows_loader(*args, **kwargs):
        kwargs['num_workers'] = 0
        return original_loader(*args, **kwargs)

    base.DataLoader = windows_loader
    base.Pool = ThreadPool
    data_utils.ProcessPoolExecutor = ThreadPoolExecutor

    data_utils.os = _OSNamespace()
    base.shutil = _ShutilNamespace()
    interface.shutil = _ShutilNamespace()
    print(
        'Windows adapter: in-process WSI reads; threaded postprocessing; '
        'original model and thresholds; lock-aware SQLite cleanup.',
        flush=True,
    )
    from inference_CoNIC import main

    # Export newly written results even when upstream cache cleanup raises.
    # Never export an unchanged old database after an unsuccessful rerun.
    import argparse
    from export_qupath import export_kongnet_db
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output_dir', type=Path)
    options, _ = parser.parse_known_args()
    output_dir = options.output_dir
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in output_dir.glob('*.db')} if output_dir else {}
    try:
        main()
    finally:
        if output_dir:
            for database in output_dir.glob('*.db'):
                signature = database.stat().st_mtime_ns, database.stat().st_size
                if before.get(database) != signature:
                    print(export_kongnet_db(database), flush=True)
