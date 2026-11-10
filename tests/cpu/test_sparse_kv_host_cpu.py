"""Host-only regression checks; no CANN, Torch or NPU is required.

Run directly: python tests/cpu/test_sparse_kv_host_cpu.py
The ACL mock checks API sequencing and ownership, not device behavior.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(os.getenv("SGL_KERNEL_CPU_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
ALLOCATOR = Path("csrc/sparsity_driven_kv_offload/shm_allocator")

ACL_HEADER = r"""
#pragma once
#include <cstddef>
#include <cstdint>
using aclError = int;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_ERROR_NONE = 0;
constexpr unsigned ACL_HOST_REG_PINNED = 1;
constexpr unsigned ACL_HOST_REG_MAPPED = 2;
aclError aclInit(const char *);
aclError aclrtGetDevice(int32_t *);
aclError aclrtSetDevice(int32_t);
aclError aclrtMallocHost(void **, size_t);
aclError aclrtFreeHost(void *);
aclError aclrtHostRegisterV2(void *, size_t, unsigned);
aclError aclrtHostGetDevicePointer(void *, void **, unsigned);
aclError aclrtHostUnregister(void *);
"""

ACL_TEST = r"""
#include "shm_allocator.h"
#include <acl/acl.h>
#include <cassert>
#include <cstdlib>
#include <unordered_set>

std::unordered_set<void *> allocated, registered;
unsigned last_flags = 0;
int register_calls = 0;
bool fail_register = false, fail_query = false, fail_unregister = false;
aclError aclInit(const char *) { return 0; }
aclError aclrtGetDevice(int32_t *device) { *device = 0; return 0; }
aclError aclrtSetDevice(int32_t) { return 0; }
aclError aclrtMallocHost(void **ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    assert(*ptr);
    allocated.insert(*ptr);
    return 0;
}
aclError aclrtFreeHost(void *ptr) {
    assert(allocated.erase(ptr) == 1);
    assert(registered.count(ptr) == 0);
    std::free(ptr);
    return 0;
}
aclError aclrtHostRegisterV2(void *ptr, size_t, unsigned flags) {
    ++register_calls;
    last_flags = flags;
    if (fail_register || ((flags & ACL_HOST_REG_PINNED) && allocated.count(ptr))) return 7;
    assert(registered.insert(ptr).second);
    return 0;
}
aclError aclrtHostGetDevicePointer(void *ptr, void **mapped, unsigned) {
    assert(registered.count(ptr));
    if (fail_query) return 8;
    *mapped = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) + 0x100000000ULL);
    return 0;
}
aclError aclrtHostUnregister(void *ptr) {
    if (fail_unregister) return 9;
    assert(registered.erase(ptr) == 1);
    return 0;
}
int main() {
    uint64_t dva = 0, reused = 0;
    assert(shm_create_and_register("owned", 0, 4096, 1, &dva) == 0);
    assert(last_flags == ACL_HOST_REG_MAPPED);
    assert(dva != reinterpret_cast<uint64_t>(shm_get_host_ptr("owned")));
    const auto calls = register_calls;
    assert(shm_create_and_register("owned", 0, 4096, 1, &reused) == 0);
    assert(reused == dva && register_calls == calls);
    assert(shm_free_by_name("owned", 0) == 0);
    assert(allocated.empty() && registered.empty());

    void *host = nullptr;
    assert(shm_create("split", 4096, &host) == 0);
    assert(shm_register("split", 0, host, 4096, &dva) == 0);
    assert(last_flags == ACL_HOST_REG_MAPPED);
    assert(shm_free_all(0) == 0 && allocated.empty());

    host = std::malloc(4096);
    assert(shm_register("external", 0, host, 4096, &dva) == 0);
    assert(shm_free_by_name("external", 0) == 0);
    std::free(host); // Allocator must not free caller-owned memory.

    fail_query = true;
    assert(shm_create_and_register("query-fail", 0, 4096, 1, &dva) == 8);
    assert(!shm_get_host_ptr("query-fail") && allocated.empty() && registered.empty());
    fail_query = false;
    fail_register = true;
    assert(shm_create_and_register("register-fail", 0, 4096, 1, &dva) == 7);
    assert(!shm_get_host_ptr("register-fail") && allocated.empty());
    fail_register = false;

    assert(shm_create_and_register("retry", 0, 4096, 1, &dva) == 0);
    fail_unregister = true;
    assert(shm_free_all(0) == -3 && shm_get_host_ptr("retry"));
    fail_unregister = false;
    assert(shm_free_all(0) == 0 && allocated.empty() && registered.empty());
}
"""


@unittest.skipUnless(shutil.which("g++"), "requires a host C++ compiler")
class TestSparseKVHost(unittest.TestCase):
    def preprocess(self, *macros):
        return subprocess.check_output(
            [
                "g++", "-E", "-P", "-x", "c++",
                *[f"-D{name}" for name in macros],
                str(ROOT / "include/sgl_kenel_npu_ops.h"),
            ],
            text=True,
        )

    def test_a5_declarations(self):
        header = self.preprocess("SGL_KERNEL_ENABLE_SPARSE_KV_IO")
        for name in ("unidex_copy", "shm_allocator_create_and_register", "shm_allocator_free_all"):
            self.assertIn(f"{name}(", header)
        self.assertNotIn("slot_map_lookup(", header)

    def test_a3_declarations(self):
        header = self.preprocess(
            "SGL_KERNEL_ENABLE_SPARSE_KV_IO", "SGL_KERNEL_ENABLE_A3_ONLY_OPS"
        )
        for name in (
            "unidex_copy", "shm_allocator_create_and_register",
            "shm_allocator_free_all", "slot_map_lookup",
        ):
            self.assertIn(f"{name}(", header)

    def test_disabled_declarations(self):
        header = self.preprocess()
        for name in (
            "unidex_copy", "shm_allocator_create_and_register",
            "shm_allocator_free_all", "slot_map_lookup",
        ):
            self.assertNotIn(f"{name}(", header)

    def test_a5_allocator_with_acl_mock(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "acl").mkdir()
            (tmp / "acl/acl.h").write_text(ACL_HEADER)
            (tmp / "test.cpp").write_text(ACL_TEST)
            subprocess.run(
                ["g++", "-std=c++17", "-Wall", "-Wextra", "-pthread",
                 "-DSGL_KERNEL_ARCH_35", "-I", str(tmp), "-I", str(ROOT / ALLOCATOR),
                 str(ROOT / ALLOCATOR / "shm_allocator.cpp"), str(tmp / "test.cpp"),
                 "-o", str(tmp / "test")], check=True
            )
            subprocess.run([str(tmp / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
