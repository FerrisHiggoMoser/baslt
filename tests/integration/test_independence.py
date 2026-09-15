"""The verifier must not reach compiler modules through any chain of imports.

The static check follows plain, relative, function-level and TYPE_CHECKING imports, import_module/__import__/
run_module calls (positional or keyword, absolute or relative, literal or with a constant prefix), and attribute
access that goes through a package's lazy `__getattr__` table. A dynamic import it cannot resolve fails the test
instead of being skipped. A runtime check complements it for module-level imports.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

pytestmark = pytest.mark.minimal

PACKAGE = "baslt"
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
# ops, reduce, plan and manifest are named in docs/architecture.md. docs/verification.md also requires the verifier
# to use its own unit table, and structure.bind re-binds the policy with that table; importing the compiler's unit
# table or policy binder would make structure.bind compare the compiler with itself.
FORBIDDEN = ("baslt.ops", "baslt.reduce", "baslt.plan", "baslt.manifest", "baslt.units", "baslt.policy")
# Callables that import the module named by their first argument.
IMPORTERS = frozenset({"import_module", "__import__", "run_module"})

LazyTable = dict[str, tuple[str, ...]]


def index_modules(src_root: Path, package: str) -> dict[str, Path]:
    """Map every dotted module name under `src_root/package` to its file."""
    modules: dict[str, Path] = {}
    for path in sorted((src_root / package).rglob("*.py")):
        parts = list(path.relative_to(src_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def _package_of(module: str, path: Path) -> str:
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def resolve_name(name: str, package: str, where: str) -> str:
    """Absolute module name for `name`, resolving leading dots against `package` the way importlib does."""
    if not name.startswith("."):
        return name
    level = len(name) - len(name.lstrip("."))
    bits = package.rsplit(".", level - 1) if package else []
    if len(bits) < level:
        raise AssertionError(f"{where}: relative import beyond top-level package (level {level})")
    rest = name[level:]
    return f"{bits[0]}.{rest}" if rest else bits[0]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def lazy_table(module: str, path: Path, tree: ast.Module) -> LazyTable | None:
    """Attribute -> possible module names for a module that resolves attributes in a module-level `__getattr__`.

    The table is every module-level dict literal that `__getattr__` refers to and whose entries map a string to a
    string or to a tuple/list of strings; every string in an entry is treated as a module it may import. Returns
    None when the module has no `__getattr__` or no such literal table, so dynamic imports there stay unexplained.
    """
    getattrs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"
    ]
    if not getattrs:
        return None
    used = {node.id for func in getattrs for node in ast.walk(func) if isinstance(node, ast.Name)}
    package = _package_of(module, path)
    table: LazyTable | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id in used and isinstance(value, ast.Dict)):
            continue
        entries: LazyTable = {}
        for key, entry in zip(value.keys, value.values):
            items = entry.elts if isinstance(entry, (ast.Tuple, ast.List)) else [entry]
            if not _is_str(key) or not items or not all(_is_str(item) for item in items):
                break
            names = [item.value for item in items]
            entries[key.value] = tuple(
                resolve_name(name, package, str(path)) if name.strip(".") else name for name in names
            )
        else:
            table = {**(table or {}), **entries}
    return table


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _string_head(node: ast.expr | None) -> tuple[str, bool]:
    """(text, complete) for a string expression; an incomplete result is only a known prefix."""
    if _is_str(node):
        return node.value, True
    if isinstance(node, ast.JoinedStr):
        head = ""
        for part in node.values:
            if not _is_str(part):
                return head, False
            head += part.value
        return head, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, left_complete = _string_head(node.left)
        if not left_complete:
            return left, False
        right, right_complete = _string_head(node.right)
        return left + right, right_complete
    return "", False


def _argument(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    if len(call.args) > index:
        return call.args[index]
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def importer_targets(
    call: ast.Call, func_name: str, module: str, package: str, known: Iterable[str], where: str
) -> list[str] | None:
    """Module names an import_module/__import__/run_module call may import, or None if it cannot be resolved."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        return None
    if func_name == "run_module":
        name_node, base, fromlist = _argument(call, 0, "mod_name"), "", None
    elif func_name == "__import__":
        name_node, base, fromlist = _argument(call, 0, "name"), package, _argument(call, 3, "fromlist")
        level_node = _argument(call, 4, "level")
        if level_node is not None and not (
            isinstance(level_node, ast.Constant) and type(level_node.value) is int and level_node.value >= 0
        ):
            return None
        level = 0 if level_node is None else level_node.value
    else:
        name_node, fromlist = _argument(call, 0, "name"), None
        package_node = _argument(call, 1, "package")
        if _is_str(package_node):
            base = package_node.value
        elif isinstance(package_node, ast.Name) and package_node.id in ("__package__", "__name__"):
            base = package if package_node.id == "__package__" else module
        else:
            base = None
    text, complete = _string_head(name_node)
    if func_name == "__import__" and level:
        text = "." * level + text
    if not text.strip("."):
        return None  # nothing but dots (or nothing) is known about the name
    if text.startswith("."):
        if base is None:
            return None
        text = resolve_name(text, base, where)
    if not complete:
        return sorted(name for name in known if name.startswith(text))
    targets = [text]
    if fromlist is not None and not (isinstance(fromlist, ast.Constant) and fromlist.value is None):
        items = fromlist.elts if isinstance(fromlist, (ast.Tuple, ast.List)) else None
        if items is not None and all(_is_str(item) and item.value != "*" for item in items):
            targets.extend(f"{text}.{item.value}" for item in items)
        else:  # computed or star fromlist: any submodule
            targets.extend(name for name in known if name.startswith(text + "."))
    return targets


def imported_names(
    module: str,
    path: Path,
    *,
    known: Iterable[str] = (),
    table_of: Callable[[str], LazyTable] | None = None,
) -> list[str]:
    """Every absolute module name a file may import, including lazy, dynamic and TYPE_CHECKING imports.

    `known` lists module names used to expand dynamic imports with a constant prefix; `table_of` returns the lazy
    `__getattr__` table of another module (empty when it has none).
    """
    tree = _parse(path)
    known = tuple(known)
    table_of = table_of or (lambda _name: {})
    package = _package_of(module, path)
    where = str(path)
    names: list[str] = []
    importers = set(IMPORTERS)
    bindings: dict[str, set[str]] = {}

    def bind(alias: str, target: str) -> None:
        bindings.setdefault(alias, set()).add(target)

    # Static imports, and the local names they bind to modules.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
                if alias.asname:
                    bind(alias.asname, alias.name)
                else:
                    top = alias.name.partition(".")[0]
                    bind(top, top)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = resolve_name("." * node.level + (node.module or ""), package, where)
            else:
                base = node.module or ""
            names.append(base)
            table = table_of(base)
            for alias in node.names:
                if alias.name == "*":
                    names.extend(target for targets in table.values() for target in targets)
                    continue
                # `from pkg import name` may import the submodule pkg.name or go through pkg.__getattr__
                names.append(f"{base}.{alias.name}")
                names.extend(table.get(alias.name, ()))
                bind(alias.asname or alias.name, f"{base}.{alias.name}")
                if alias.name in IMPORTERS:
                    importers.add(alias.asname or alias.name)

    def resolved_call(node: ast.expr) -> list[str]:
        if isinstance(node, ast.Call) and _call_name(node.func) in importers:
            targets = importer_targets(node, _call_name(node.func), module, package, known, where)
            return targets or []
        return []

    def modules_of(node: ast.expr) -> set[str]:
        if isinstance(node, ast.Name):
            return bindings.get(node.id, set())
        return set(resolved_call(node))

    # `name = import_module("pkg")` binds a module too.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target_module in resolved_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bind(target.id, target_module)

    # Dynamic imports in a module-level __getattr__ are explained by the module's own lazy table, which importers
    # of this module are charged for when they touch a listed attribute.
    explained: set[int] = set()
    if lazy_table(module, path, tree) is not None:
        for func in tree.body:
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and func.name == "__getattr__":
                explained.update(id(node) for node in ast.walk(func))

    def charge_attributes(owners: Iterable[str], attrs: list[str]) -> None:
        for owner in owners:
            for attr in attrs:
                table = table_of(owner)
                if attr == "__getattr__":
                    names.extend(target for targets in table.values() for target in targets)
                else:
                    names.extend(table.get(attr, ()))
                owner = f"{owner}.{attr}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _call_name(node.func)
            if func_name in importers:
                targets = importer_targets(node, func_name, module, package, known, where)
                if targets is None:
                    if id(node) not in explained:
                        raise AssertionError(
                            f"{where}:{node.lineno}: dynamic import cannot be resolved statically: "
                            f"{ast.unparse(node)}"
                        )
                    continue
                names.extend(targets)
            elif isinstance(node.func, ast.Name) and func_name in ("getattr", "hasattr") and node.args:
                key = node.args[1] if len(node.args) > 1 else None
                for owner in modules_of(node.args[0]):
                    if _is_str(key):
                        charge_attributes([owner], [key.value])
                    else:
                        charge_attributes([owner], ["__getattr__"])
        elif isinstance(node, ast.Attribute):
            attrs: list[str] = []
            base_node: ast.expr = node
            while isinstance(base_node, ast.Attribute):
                attrs.append(base_node.attr)
                base_node = base_node.value
            attrs.reverse()
            charge_attributes(modules_of(base_node), attrs)
    return names


def _with_parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def reachable_from(src_root: Path, package: str, start_prefix: str) -> dict[str, str | None]:
    """Transitive closure of in-package modules reachable from every module under `start_prefix`.

    Returns module name -> the module that first imported it (None for start modules), so a
    failure can print the import chain. Importing a.b.c also executes a and a.b, so parents count.
    """
    modules = index_modules(src_root, package)
    tables: dict[str, LazyTable] = {}

    def table_of(name: str) -> LazyTable:
        if name not in tables:
            path = modules.get(name)
            tables[name] = (lazy_table(name, path, _parse(path)) if path is not None else None) or {}
        return tables[name]

    starts = [m for m in modules if m == start_prefix or m.startswith(start_prefix + ".")]
    via: dict[str, str | None] = {}
    queue: deque[str] = deque()

    def visit(name: str, parent: str | None) -> None:
        for candidate in _with_parents(name):
            if candidate not in via:
                via[candidate] = parent
                if candidate in modules:
                    queue.append(candidate)

    for start in starts:
        visit(start, None)
    while queue:
        current = queue.popleft()
        for name in imported_names(current, modules[current], known=modules, table_of=table_of):
            if name == package or name.startswith(package + "."):
                visit(name, current)
    return via


def forbidden_hits(via: dict[str, str | None], forbidden: tuple[str, ...]) -> list[str]:
    hits = []
    for name in sorted(via):
        if any(name == f or name.startswith(f + ".") for f in forbidden):
            chain = [name]
            while via[chain[-1]] is not None:
                chain.append(via[chain[-1]])
            hits.append(" <- ".join(chain))
    return hits


def _verify_modules() -> list[str]:
    modules = index_modules(SRC_ROOT, PACKAGE)
    return sorted(m for m in modules if m == f"{PACKAGE}.verify" or m.startswith(f"{PACKAGE}.verify."))


def test_verify_does_not_reach_compiler_modules():
    verify_dir = SRC_ROOT / PACKAGE / "verify"
    assert (verify_dir / "__init__.py").is_file()
    assert (verify_dir / "si_table.py").is_file()
    via = reachable_from(SRC_ROOT, PACKAGE, f"{PACKAGE}.verify")
    assert f"{PACKAGE}.verify.si_table" in via
    hits = forbidden_hits(via, FORBIDDEN)
    assert not hits, "verify reaches compiler modules:\n" + "\n".join(hits)


def test_every_verify_file_parses():
    modules = index_modules(SRC_ROOT, PACKAGE)
    verify_modules = _verify_modules()
    assert verify_modules
    for name in verify_modules:
        imported_names(name, modules[name], known=modules)


def test_importing_verify_loads_no_compiler_module():
    code = (
        "import importlib, sys\n"
        f"for name in {_verify_modules()!r}:\n"
        "    importlib.import_module(name)\n"
        "print('\\n'.join(sorted(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    loaded = out.stdout.split()
    assert "baslt.verify.si_table" in loaded
    bad = [name for name in loaded if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)]
    assert not bad, f"importing baslt.verify loaded compiler modules: {bad}"


# Self-tests of the checker on a synthetic package tree.


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from __future__ import annotations\n" + text, encoding="utf-8")


@pytest.fixture
def fake_tree(tmp_path: Path) -> Path:
    for rel in (
        "baslt/__init__.py",
        "baslt/ops/__init__.py",
        "baslt/ops/crossing.py",
        "baslt/reduce.py",
        "baslt/plan/__init__.py",
        "baslt/manifest.py",
        "baslt/units.py",
        "baslt/policy/__init__.py",
        "baslt/policy/bind.py",
        "baslt/planner_notes.py",
        "baslt/verify/__init__.py",
        "baslt/util/__init__.py",
        "baslt/util/helper.py",
    ):
        _write(tmp_path, rel)
    return tmp_path


def _hits(root: Path) -> list[str]:
    return forbidden_hits(reachable_from(root, "baslt", "baslt.verify"), FORBIDDEN)


def _assert_hit(root: Path, expected: str) -> None:
    hits = _hits(root)
    assert any(hit.split(" <- ")[0] == expected for hit in hits), hits


def test_checker_clean_tree(fake_tree):
    _write(fake_tree, "baslt/verify/checks.py", "import math\nfrom . import si\nfrom .si import x\n")
    _write(fake_tree, "baslt/verify/si.py", "from ..util import helper\nimport baslt.planner_notes\n")
    assert _hits(fake_tree) == []
    via = reachable_from(fake_tree, "baslt", "baslt.verify")
    assert "baslt.util.helper" in via and "baslt.planner_notes" in via


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import baslt.ops\n", "baslt.ops"),
        ("import baslt.ops.crossing as c\n", "baslt.ops.crossing"),
        ("from baslt import reduce\n", "baslt.reduce"),
        ("from baslt.plan import thing\n", "baslt.plan"),
        ("from .. import manifest\n", "baslt.manifest"),
        ("from ..ops.crossing import find\n", "baslt.ops.crossing"),
        ("def f():\n    from ..plan import x\n", "baslt.plan"),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from baslt.manifest import M\n",
            "baslt.manifest",
        ),
        # The compiler's unit table and policy binder.
        ("from baslt.units import UNITS\n", "baslt.units"),
        ("from .. import units\n", "baslt.units"),
        ("from baslt.policy import bind_policy\n", "baslt.policy"),
        ("def f():\n    from ..policy.bind import bind\n", "baslt.policy.bind"),
        # Dynamic imports.
        ("from importlib import import_module\nm = import_module('baslt.reduce')\n", "baslt.reduce"),
        ("from importlib import import_module\nm = import_module(name='baslt.reduce')\n", "baslt.reduce"),
        ("from importlib import import_module\nm = import_module('.reduce', 'baslt')\n", "baslt.reduce"),
        ("import importlib\nm = importlib.import_module('..reduce', package=__package__)\n", "baslt.reduce"),
        # against __name__ "baslt.verify.checks", three dots climb to "baslt", as importlib.util.resolve_name does
        ("import importlib\nm = importlib.import_module('...units', __name__)\n", "baslt.units"),
        ("from importlib import import_module as load\nm = load(package='baslt', name='.plan')\n", "baslt.plan"),
        ("import importlib\nm = importlib.import_module('baslt.' + 'units')\n", "baslt.units"),
        ("import importlib\nm = importlib.import_module(f'baslt.ops.{name}')\n", "baslt.ops.crossing"),
        ("import importlib\nm = importlib.import_module('baslt.pol' + name)\n", "baslt.policy.bind"),
        ("import importlib\nm = importlib.import_module(f'..pl{name}', 'baslt.verify')\n", "baslt.plan"),
        ("m = __import__('baslt.manifest')\n", "baslt.manifest"),
        ("m = __import__('baslt', fromlist=['reduce'])\n", "baslt.reduce"),
        ("m = __import__('baslt', globals(), None, names)\n", "baslt.units"),
        ("m = __import__('reduce', globals(), None, (), 2)\n", "baslt.reduce"),
        ("m = __import__(name='plan', level=2)\n", "baslt.plan"),
        ("import runpy\nrunpy.run_module('baslt.reduce')\n", "baslt.reduce"),
    ],
)
def test_checker_direct_imports(fake_tree, source, expected):
    _write(fake_tree, "baslt/verify/checks.py", source)
    _assert_hit(fake_tree, expected)


def test_checker_follows_transitive_chain(fake_tree):
    _write(fake_tree, "baslt/verify/checks.py", "from ..util.helper import h\n")
    _write(fake_tree, "baslt/util/helper.py", "from . import deeper\n")
    _write(fake_tree, "baslt/util/deeper.py", "from baslt.ops import crossing\n")
    hits = _hits(fake_tree)
    assert "baslt.ops.crossing <- baslt.util.deeper <- baslt.util.helper <- baslt.verify.checks" in hits


def test_checker_follows_chain_into_unit_table(fake_tree):
    _write(fake_tree, "baslt/verify/checks.py", "from ..util.helper import h\n")
    _write(fake_tree, "baslt/util/helper.py", "from ..units import parse_quantity\n")
    assert "baslt.units <- baslt.util.helper <- baslt.verify.checks" in _hits(fake_tree)


def test_checker_follows_package_init(fake_tree):
    _write(fake_tree, "baslt/__init__.py", "from .plan import planner\n")
    _write(fake_tree, "baslt/verify/checks.py", "")
    assert any(hit.startswith("baslt.plan") for hit in _hits(fake_tree))


def test_checker_nested_verify_subpackage(fake_tree):
    _write(fake_tree, "baslt/verify/sub/__init__.py", "from . import leaf\n")
    _write(fake_tree, "baslt/verify/sub/leaf.py", "from ...reduce import r\n")
    assert any(hit.startswith("baslt.reduce <- baslt.verify.sub.leaf") for hit in _hits(fake_tree))


def test_checker_similar_names_are_not_forbidden(fake_tree):
    _write(fake_tree, "baslt/verify/checks.py", "import baslt.planner_notes\n")
    assert _hits(fake_tree) == []


def test_checker_rejects_relative_import_beyond_top(fake_tree):
    _write(fake_tree, "baslt/verify/checks.py", "from .... import nope\n")
    with pytest.raises(AssertionError, match="beyond top-level"):
        _hits(fake_tree)


# The lazy-attribute pattern of src/baslt/__init__.py, with entries that point at compiler modules.
LAZY_INIT = """
from importlib import import_module

from ._version import __version__

# Public name -> (module, attribute). Resolved on first access.
_LAZY: dict[str, tuple[str, str]] = {
    "compile": ("baslt.plan", "compile"),
    "reduce_all": (".reduce", "run"),
    "helper": ("baslt.util.helper", "helper"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'baslt' has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
"""


@pytest.fixture
def lazy_tree(fake_tree: Path) -> Path:
    _write(fake_tree, "baslt/__init__.py", LAZY_INIT)
    _write(fake_tree, "baslt/_version.py", "__version__ = '0'\n")
    return fake_tree


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from baslt import compile\n", "baslt.plan"),
        ("from .. import reduce_all\n", "baslt.reduce"),
        ("from baslt import compile as build\n", "baslt.plan"),
        ("from baslt import *\n", "baslt.plan"),
        ("import baslt\nbaslt.compile(1)\n", "baslt.plan"),
        ("import baslt.util.helper\ndef f():\n    return baslt.reduce_all\n", "baslt.reduce"),
        ("import baslt as b\nx = getattr(b, 'compile')\n", "baslt.plan"),
        ("import baslt\nx = hasattr(baslt, 'reduce_all')\n", "baslt.reduce"),
        ("import baslt\nx = getattr(baslt, name)\n", "baslt.plan"),
        ("import baslt\nx = baslt.__getattr__(name)\n", "baslt.plan"),
        ("import importlib\nx = importlib.import_module('baslt').compile\n", "baslt.plan"),
        ("from importlib import import_module\npkg = import_module('baslt')\nx = pkg.reduce_all\n", "baslt.reduce"),
    ],
)
def test_checker_follows_lazy_getattr(lazy_tree, source, expected):
    _write(lazy_tree, "baslt/verify/checks.py", source)
    _assert_hit(lazy_tree, expected)


def test_checker_lazy_table_charges_only_touched_names(lazy_tree):
    source = "import baslt\nfrom baslt import helper\nx = baslt.helper\ny = baslt.__version__\n"
    _write(lazy_tree, "baslt/verify/checks.py", source)
    assert _hits(lazy_tree) == []
    via = reachable_from(lazy_tree, "baslt", "baslt.verify")
    assert via["baslt.util.helper"] == "baslt.verify.checks"


def test_checker_lazy_table_in_subpackage(fake_tree):
    _write(fake_tree, "baslt/util/__init__.py", LAZY_INIT.replace("from ._version import __version__", ""))
    _write(fake_tree, "baslt/verify/checks.py", "from ..util import reduce_all\n")
    # ".reduce" in baslt.util resolves to baslt.util.reduce, which does not exist; "baslt.plan" is absolute
    assert _hits(fake_tree) == []
    _write(fake_tree, "baslt/verify/checks.py", "from ..util import compile\n")
    _assert_hit(fake_tree, "baslt.plan")


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nname = 'baslt.plan'\nm = importlib.import_module(name)\n",
        "import importlib\nm = importlib.import_module('.reduce')\n",
        "import importlib\nm = importlib.import_module('.reduce', package=pkg)\n",
        "import importlib\nm = importlib.import_module(f'{root}.plan')\n",
        "import importlib\nm = importlib.import_module(*names)\n",
        "m = __import__('reduce', level=depth)\n",
        "def __getattr__(name):\n    import importlib\n    return importlib.import_module(name)\n",
    ],
)
def test_checker_rejects_unresolvable_dynamic_import(fake_tree, source):
    _write(fake_tree, "baslt/verify/checks.py", source)
    with pytest.raises(AssertionError, match="cannot be resolved statically"):
        _hits(fake_tree)


def test_checker_rejects_getattr_without_literal_table(fake_tree):
    init = (
        "from importlib import import_module\n"
        "_LAZY = dict(compile=('baslt.plan', 'compile'))\n"
        "def __getattr__(name):\n"
        "    module_name, attr = _LAZY[name]\n"
        "    return getattr(import_module(module_name), attr)\n"
    )
    _write(fake_tree, "baslt/__init__.py", init)
    _write(fake_tree, "baslt/verify/checks.py", "")
    with pytest.raises(AssertionError, match="cannot be resolved statically"):
        _hits(fake_tree)


def test_checker_recognizes_real_package_init():
    """The real baslt/__init__.py uses a lazy table the checker understands, so its dynamic import is explained."""
    path = SRC_ROOT / PACKAGE / "__init__.py"
    tree = _parse(path)
    has_getattr = any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body)
    if has_getattr:
        assert lazy_table(PACKAGE, path, tree) is not None
    imported_names(PACKAGE, path, known=index_modules(SRC_ROOT, PACKAGE))
