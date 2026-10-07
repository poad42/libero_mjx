"""Fix warp's nested-function source extraction (nested @wp.kernel / @wp.func).

``Adjoint.extract_function_source`` resolves a function's source from the code
object's ``co_firstlineno`` against the file as it is on disk *now*. When the
file is rewritten after the code object was compiled (for example
``libero_mjx`` patches ``mujoco_warp/_src/render.py`` after ``mujoco_warp`` has
already been imported), the stale line number lands on a different function.
The fast path notices the name mismatch and falls back to
``inspect.getsourcelines``, but the fallback never validates the result, so it
silently returns the wrong ``FunctionDef``. ``Adjoint.__init__`` then sets
``adj.fun_name`` from that wrong node and ``ModuleBuilder.build_kernel`` rejects
the enclosing function's ``return`` as a kernel value return::

    WarpCodegenTypeError: '_build_megakernel__locals___render_megakernel':
    Warp kernels cannot return values.

This patch makes ``extract_function_source`` enforce its documented contract --
``tree.body[0]`` is the ``FunctionDef`` whose ``name`` is the code object's
``co_name`` -- on the slow path too. When the line-number slice is wrong, it
recovers the definition by name (preferring a ``co_qualname`` match) from the
current file.

Idempotent: running it twice changes nothing and reports skipped.

Usage:
  python patch_warp_nested_kernel.py <warp source root>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

OLD_TAIL = '''        source, fun_lineno = Adjoint._inspect_extract_function_source(func)
        dedented = textwrap.dedent(source)
        return dedented, fun_lineno, ast.parse(dedented)
'''

NEW_TAIL = '''        source, fun_lineno = Adjoint._inspect_extract_function_source(func)
        dedented = textwrap.dedent(source)
        try:
            tree = ast.parse(dedented)
        except SyntaxError:
            tree = None
        if code is not None and not (
            tree is not None
            and tree.body
            and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
            and tree.body[0].name == code.co_name
        ):
            recovered = Adjoint._recover_function_source(code)
            if recovered is not None:
                return recovered
        if tree is None:
            tree = ast.parse(dedented)
        return dedented, fun_lineno, tree
'''

OLD_INSERT = '''        return "".join(lines[start:end]), code.co_firstlineno

    # generate function ssa form and adjoint
'''

NEW_METHODS = '''        return "".join(lines[start:end]), code.co_firstlineno

    @staticmethod
    def _iter_qualified_function_defs(node, prefix=""):
        """Yield ``(FunctionDef, qualified_name)`` for every definition in ``node``."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                yield child, qualname
                yield from Adjoint._iter_qualified_function_defs(child, f"{qualname}.<locals>")
            elif isinstance(child, ast.ClassDef):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                yield from Adjoint._iter_qualified_function_defs(child, qualname)
            else:
                yield from Adjoint._iter_qualified_function_defs(child, prefix)

    @staticmethod
    def _recover_function_source(code: types.CodeType) -> tuple[str, int, ast.Module] | None:
        """Re-extract a definition by name from the current file.

        Used when the line-number based slice starts with the wrong function,
        which happens when the source file was rewritten after ``code`` was
        compiled. Prefers a ``co_qualname`` match, then the definition closest
        to ``co_firstlineno``. Returns ``None`` when no matching definition is
        found, leaving the original (unvalidated) result in place.
        """
        try:
            source = "".join(linecache.getlines(code.co_filename))
        except Exception:
            return None
        if not source:
            return None
        try:
            module = ast.parse(source)
        except SyntaxError:
            return None
        qualname = getattr(code, "co_qualname", None)
        exact = None
        nearest = None
        nearest_dist = None
        for node, node_qualname in Adjoint._iter_qualified_function_defs(module):
            if node.name != code.co_name:
                continue
            if qualname is not None and node_qualname == qualname:
                exact = node
                break
            dist = abs(node.lineno - code.co_firstlineno)
            if nearest is None or dist < nearest_dist:
                nearest = node
                nearest_dist = dist
        best = exact if exact is not None else nearest
        if best is None:
            return None
        lines = source.splitlines(keepends=True)
        start = min([best.lineno] + [d.lineno for d in getattr(best, "decorator_list", [])])
        segment = "".join(lines[start - 1:best.end_lineno])
        dedented = textwrap.dedent(segment)
        try:
            tree = ast.parse(dedented)
        except SyntaxError:
            return None
        if not (
            tree.body
            and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
            and tree.body[0].name == code.co_name
        ):
            return None
        return dedented, start, tree

    # generate function ssa form and adjoint
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
    shutil.copyfile(path, str(path) + ".orig")
    for old, new in patches:
        text = text.replace(old, new, 1)
    open(path, "w").write(text)
    print("patched:", path)
    return True


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_warp_nested_kernel.py <warp source root>")
    root = Path(sys.argv[1])
    changed = patch_file(
        root / "warp" / "_src" / "codegen.py",
        [(OLD_TAIL, NEW_TAIL), (OLD_INSERT, NEW_METHODS)],
    )
    print("changed:", sum(1 for c in [changed] if c))


if __name__ == "__main__":
    main()
