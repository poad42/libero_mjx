"""Port warp's CPU JIT build to a modern LLVM (standalone warp-clang).

This is required only on the ROCm path, where warp is built from source against
the LLVM shipped inside the pip ``rocm-sdk``. Three source-level fixes are
needed for ``build_lib.py --standalone --llvm-path=<sdk LLVM>``:

1. LLVM 23 changed the ORC ``ObjectLinkingLayerCreator`` signature.
2. ``build_llvm.py`` links every static ``.a`` it finds, including ``libc++abi.a``
   which collides with gcc's ``libstdc++.a`` on ``__cxa_*``.
3. Warp's host C++ uses the pre-C++11 libstdc++ ABI while the SDK LLVM uses the
   cxx11 ABI, so ``warp-clang.so`` links with an undefined symbol.

The third fix is what ``WARP_GLIBCXX_USE_CXX11_ABI=1`` selects at build time.

Idempotent: running it twice changes nothing and reports skipped.

Usage:
  python patch_warp_llvm23.py <warp source root>
"""

from __future__ import annotations

import os
import shutil
import sys

LEGACY_OLD = """        builder.setObjectLinkingLayerCreator(
#if LLVM_VERSION_MAJOR >= 21
            [](llvm::orc::ExecutionSession& session)
#else
            [](llvm::orc::ExecutionSession& session, const llvm::Triple& triple)
#endif
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
#if LLVM_VERSION_MAJOR >= 21
                auto get_memory_manager = [](const llvm::MemoryBuffer&) {
"""

LEGACY_NEW = """        builder.setObjectLinkingLayerCreator(
#if LLVM_VERSION_MAJOR >= 23
            [](llvm::orc::ExecutionSession& session, llvm::jitlink::JITLinkMemoryManager& memMgr)
#elif LLVM_VERSION_MAJOR >= 21
            [](llvm::orc::ExecutionSession& session)
#else
            [](llvm::orc::ExecutionSession& session, const llvm::Triple& triple)
#endif
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
#if LLVM_VERSION_MAJOR >= 21
                auto get_memory_manager = [](const llvm::MemoryBuffer&) {
"""

DEFAULT_OLD = """        builder.setObjectLinkingLayerCreator(
#if LLVM_VERSION_MAJOR >= 21
            [](llvm::orc::ExecutionSession& session)
#else
            [](llvm::orc::ExecutionSession& session, const llvm::Triple& triple)
#endif
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);
"""

DEFAULT_NEW = """        builder.setObjectLinkingLayerCreator(
#if LLVM_VERSION_MAJOR >= 23
            [](llvm::orc::ExecutionSession& session, llvm::jitlink::JITLinkMemoryManager& memMgr)
#elif LLVM_VERSION_MAJOR >= 21
            [](llvm::orc::ExecutionSession& session)
#else
            [](llvm::orc::ExecutionSession& session, const llvm::Triple& triple)
#endif
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
#if LLVM_VERSION_MAJOR >= 23
                auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session, memMgr);
#else
                auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);
#endif
"""


LIBS_OLD = '''            libs = [f"-l{lib[3:-2]}" for lib in libs if os.path.splitext(lib)[1] == ".a"]
'''

LIBS_NEW = '''            _skip_libs = {"libc++abi.a"}
            libs = [
                f"-l{lib[3:-2]}"
                for lib in libs
                if os.path.splitext(lib)[1] == ".a" and lib not in _skip_libs
            ]
'''

EXTRA_OLD = '''            libs.append("-lpthread")
            libs.append("-ldl")
            if sys.platform != "darwin":
                libs.append("-lrt")
'''

EXTRA_NEW = '''            libs.append("-lpthread")
            libs.append("-ldl")
            if sys.platform != "darwin":
                libs.append("-lrt")
                libs.append("-lzstd")
                libs.append("-lz")
'''

ABI_OLD = '''        cpp_flags = f'-Werror -Wuninitialized {version} --std=c++17 -fno-rtti -D{cuda_enabled} -D{mathdx_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -D_GLIBCXX_USE_CXX11_ABI=0 -I"{native_dir}" {includes} '
'''

ABI_NEW = '''        _is_clang_lib = "warp-clang" in os.path.basename(dll_path)
        _cxx11_abi = os.environ.get("WARP_GLIBCXX_USE_CXX11_ABI", "0") if _is_clang_lib else "0"
        cpp_flags = f'-Werror -Wuninitialized {version} --std=c++17 -fno-rtti -D{cuda_enabled} -D{mathdx_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -D_GLIBCXX_USE_CXX11_ABI={_cxx11_abi} -I"{native_dir}" {includes} '
'''


def patch_file(path, patches):
    text = open(path).read()
    originals = [old for old, _ in patches]
    news = [new for _, new in patches]
    if all(new in text for new in news):
        print("already patched:", path)
        return False
    if not all(old in text for old in originals):
        missing = [i for i, old in enumerate(originals) if old not in text]
        raise SystemExit(f"patch rejected, pattern not found in {path}: index {missing}")
    shutil.copyfile(path, path + ".orig")
    for old, new in patches:
        text = text.replace(old, new, 1)
    open(path, "w").write(text)
    print("patched:", path)
    return True


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_warp_llvm23.py <warp source root>")
    root = sys.argv[1]
    changed = []
    changed.append(patch_file(os.path.join(root, "warp/native/clang/clang.cpp"),
                              [(LEGACY_OLD, LEGACY_NEW), (DEFAULT_OLD, DEFAULT_NEW)]))
    changed.append(patch_file(os.path.join(root, "build_llvm.py"),
                              [(LIBS_OLD, LIBS_NEW), (EXTRA_OLD, EXTRA_NEW)]))
    changed.append(patch_file(os.path.join(root, "warp/_src/build_dll.py"),
                              [(ABI_OLD, ABI_NEW)]))
    print("changed:", sum(1 for c in changed if c))


if __name__ == "__main__":
    main()
