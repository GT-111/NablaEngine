"""CPU multiprocessing checks for shared build-cache compilation and publication.

PYTHONPATH=src python scripts/build_cache_check.py
A small executable stands in for nvcc; no CUDA compilation or GPU is needed.
"""

import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nablaengine import jax_ffi, torch_ffi
from model_generation_check import hand_model


def library_worker(cache, compiler, tag, misses, results):
    original_exists, first = Path.exists, True

    def checked_exists(path):
        nonlocal first
        exists = original_exists(path)
        if path.suffix == ".so" and first and not exists:
            first = False
            misses.wait(timeout=10)  # force every caller to observe the same initial cache miss
        return exists

    with patch.dict(os.environ, DK_CACHE_DIR=str(cache)), patch.object(jax_ffi.cuda, "nvcc", return_value=str(compiler)), \
            patch.object(Path, "exists", checked_exists):
        try:
            path, built, key = jax_ffi._build({"unit.cu": tag}, {"unit.cu": []}, "sm_80", None,
                                            np.zeros(1, np.float32))
            results.put(dict(status="ok", built=built, key=key, path=str(path)))
        except Exception as error:  # noqa: BLE001 (report a failed subprocess to the parent test)
            results.put(dict(status="error", message=str(error)))


def publish_artifact(cache, connection=None):
    model = hand_model(1, 1)
    cache = Path(cache)
    library = cache / "libdk_test.so"
    consts = np.arange(16384, dtype=np.float32)
    plan = dict(units={"unit.cu": ""}, defines={"unit.cu": []}, unit_arch={}, consts=consts,
                arch_flags=dict(forward=[], adjoint=[]), forward_unit="dk_fwd", forward_defines=[])
    original_save = np.save

    def slow_write(path, data, phase):
        with Path(path).open("wb") as stream:
            mid = len(data) // 2
            stream.write(data[:mid])
            stream.flush()
            connection.send(phase)
            connection.recv()
            stream.write(data[mid:])

    def save(path, array):
        buffer = io.BytesIO()
        original_save(buffer, array)
        slow_write(path, buffer.getvalue(), "constants")

    def write(path, text, *args, **kwargs):
        phase = "current" if path.name.endswith("current.json") else "metadata"
        slow_write(path, text.encode(), phase)
        return len(text)

    with patch.dict(os.environ, DK_CACHE_DIR=str(cache)), \
            patch.object(jax_ffi, "build_plan", return_value=plan), \
            patch.object(jax_ffi, "_build", return_value=(library, False, "test")), \
            patch.object(torch_ffi, "sources_sha256", return_value="test"):
        if connection is None:
            torch_ffi.build("sm_80", model=model, foot_pairs=[], com_body_id=1)
        else:
            with patch.object(np, "save", save), patch.object(Path, "write_text", write):
                torch_ffi.build("sm_80", model=model, foot_pairs=[], com_body_id=1)
            connection.send("done")


class BuildCacheCheck(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="nabla-build-cache-cpu-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache, self.log = self.root / "cache", self.root / "compile.log"
        self.cache.mkdir()
        self.context = mp.get_context("fork")  # the package supports Linux, matching the actual rank launch
        self.compiler = self.root / "nvcc"
        self.compiler.write_text(f'''#!{sys.executable}
import os
import json
from pathlib import Path
import sys
import time

with Path({str(self.root / 'commands.jsonl')!r}).open("a") as log:
    log.write(json.dumps(sys.argv[1:])+"\\n")
if "--version" in sys.argv:
    print("CPU fake nvcc 12.9")
    raise SystemExit(0)
root=Path({str(self.root)!r})
with (root/"working_dirs.jsonl").open("a") as log:
    log.write(json.dumps(str(Path.cwd()))+"\\n")
output=Path(sys.argv[sys.argv.index("-o")+1])
if "-c" in sys.argv:
    tag=Path(sys.argv[-1]).read_text()
    with (root/"compile.log").open("a") as log:
        log.write(tag+"\\n")
    if tag=="FAIL_ONCE" and not (root/"failed-once").exists():
        (root/"failed-once").touch()
        print("intentional compiler failure")
        raise SystemExit(1)
    if tag.startswith("PARALLEL_"):
        (root/tag).touch()
        deadline=time.monotonic()+5
        while not all((root/name).exists() for name in ("PARALLEL_A","PARALLEL_B")):
            if time.monotonic()>deadline:
                raise SystemExit("different cache keys were serialized")
            time.sleep(.01)
output.write_bytes(b"complete")
''')
        self.compiler.chmod(0o755)

    def builds(self, tags):
        barrier, results = self.context.Barrier(len(tags)), self.context.Queue()
        processes = [self.context.Process(target=library_worker,
            args=(self.cache, self.compiler, tag, barrier, results)) for tag in tags]
        for process in processes:
            process.start()
        try:
            output = [results.get(timeout=15) for _ in tags]
            for process in processes:
                process.join(timeout=2)
                self.assertEqual(process.exitcode, 0)
            return output
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()

    def test_same_key_compiles_once_for_four_concurrent_misses(self):
        result = self.builds(["SAME"] * 4)
        self.assertEqual([row["status"] for row in result], ["ok"] * 4)
        self.assertEqual(sum(row["built"] for row in result), 1)
        self.assertEqual(len({row["key"] for row in result}), 1)
        self.assertEqual(self.log.read_text().splitlines(), ["SAME"])

    def test_different_keys_compile_in_parallel(self):
        result = self.builds(["PARALLEL_A", "PARALLEL_B"])
        self.assertEqual([row["status"] for row in result], ["ok", "ok"])
        self.assertTrue(all(row["built"] for row in result))
        self.assertEqual(len({row["key"] for row in result}), 2)

    def test_compile_failure_releases_lock_for_waiting_retry(self):
        result = self.builds(["FAIL_ONCE"] * 2)
        self.assertEqual(sorted(row["status"] for row in result), ["error", "ok"])
        self.assertIn("intentional compiler failure", next(row["message"] for row in result if row["status"] == "error"))
        self.assertEqual(self.log.read_text().splitlines(), ["FAIL_ONCE", "FAIL_ONCE"])
        self.assertEqual(list(self.cache.glob("build_*")), [])

    def build_units(self, count=3):
        units = {f"unit{index}.cu": f"UNIT_{index}" for index in range(count)}
        with patch.dict(os.environ, DK_CACHE_DIR=str(self.cache)), \
                patch.object(jax_ffi.cuda, "nvcc", return_value=str(self.compiler)):
            return jax_ffi._build(units, {name: [] for name in units}, "sm_90", None,
                                  np.zeros(1, np.float32))

    def test_max_jobs_reaches_the_actual_compilation_pool(self):
        with patch.dict(os.environ, MAX_JOBS="2", DK_COMPILE_THREADS="1"), \
                patch.object(jax_ffi, "ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pool:
            self.build_units()
        self.assertEqual(pool.call_args.kwargs["max_workers"], 2)
        directories = [Path(json.loads(line)) for line in (self.root / "working_dirs.jsonl").read_text().splitlines()]
        self.assertEqual(len(directories), 4)  # three compile jobs and the final link
        self.assertTrue(all(directory.parent == self.cache for directory in directories))

    def test_affinity_and_cgroup_bound_nested_compiler_parallelism(self):
        original = Path.read_text

        def quota(path, *args, **kwargs):
            return "4800000 100000" if str(path) == "/sys/fs/cgroup/cpu.max" else original(path, *args, **kwargs)

        with patch.dict(os.environ, MAX_JOBS="48", DK_COMPILE_THREADS="4"), \
                patch.object(os, "sched_getaffinity", return_value=set(range(224))), \
                patch.object(Path, "read_text", quota), \
                patch.object(jax_ffi, "ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pool:
            self.build_units(13)
        self.assertEqual(pool.call_args.kwargs["max_workers"], 12)
        commands = [json.loads(line) for line in (self.root / "commands.jsonl").read_text().splitlines()]
        self.assertTrue(all("--split-compile=4" in command for command in commands if "-c" in command))
        self.assertTrue(all("--split-compile=4" not in command for command in commands if "-shared" in command))

    def test_job_count_keeps_cache_but_split_compile_is_a_distinct_artifact(self):
        with patch.dict(os.environ, MAX_JOBS="1", DK_COMPILE_THREADS="1"):
            first = self.build_units(1)
        with patch.dict(os.environ, MAX_JOBS="2", DK_COMPILE_THREADS="1"):
            second = self.build_units(1)
        with patch.dict(os.environ, MAX_JOBS="1", DK_COMPILE_THREADS="2"):
            third = self.build_units(1)
        self.assertTrue(first[1])
        self.assertFalse(second[1])
        self.assertEqual(first[2], second[2])
        self.assertTrue(third[1])
        self.assertNotEqual(first[2], third[2])

    def test_readers_never_see_partial_constants_metadata_or_current_pointer(self):
        publish_artifact(self.cache)
        parent, child = self.context.Pipe()
        process = self.context.Process(target=publish_artifact, args=(self.cache, child))
        process.start()
        try:
            for phase in ("constants", "metadata", "current"):
                self.assertTrue(parent.poll(10), phase)
                self.assertEqual(parent.recv(), phase)
                try:
                    np.testing.assert_array_equal(np.load(self.cache / "libdk_test.npy"), np.arange(16384, dtype=np.float32))
                    self.assertEqual(json.loads((self.cache / "libdk_test.json").read_text())["key"], "test")
                    self.assertEqual(json.loads((self.cache / "torch_sm_80_current.json").read_text()),
                                     {"artifact": "libdk_test.json"})
                finally:
                    parent.send("continue")
            self.assertTrue(parent.poll(10))
            self.assertEqual(parent.recv(), "done")
            process.join(timeout=2)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    unittest.main()
