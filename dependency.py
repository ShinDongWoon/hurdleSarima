import os
import re
import ast
import subprocess
import sys


def read_pyproject_dependencies(path: str = "pyproject.toml") -> list:
    """Return a list of dependencies from pyproject.toml."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - fallback when tomllib missing
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            tomllib = None

    if tomllib is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("dependencies", [])

    # Fallback: naive parse of dependencies list
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"dependencies\s*=\s*(\[[^\]]*\])", content, re.MULTILINE)
    if match:
        deps_str = re.sub(r"#.*", "", match.group(1))
        try:
            return ast.literal_eval(deps_str)
        except Exception:
            pass
    return []


def install_dependencies():
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        return

    deps = read_pyproject_dependencies()
    if not deps:
        raise FileNotFoundError("No requirements.txt or project.dependencies found")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])


if __name__ == "__main__":
    install_dependencies()
