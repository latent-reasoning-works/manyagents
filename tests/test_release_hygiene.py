"""Keep retired internal identifiers out of public source and distributions."""

import hashlib
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOKEN_HASHES = {
    "9dfe5c985dbba48aeef7b1af1cf2f5f3564b304d1aac0b91159efe060adcc808",
}


def test_public_tree_has_no_retired_identifiers():
    if (ROOT / ".git").exists():
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        ).decode().split("\0")
        paths = [ROOT / name for name in names if name]
    else:
        # An extracted sdist can contain a local venv or build output after use.
        # Scan the shipped roots, not downloaded third-party dependencies.
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        includes = project['tool']['hatch']['build']['targets']['sdist']['only-include']
        paths = []
        for name in includes:
            path = ROOT / name
            paths.extend(path.rglob("*") if path.is_dir() else [path])

    affected_files = 0
    for path in paths:
        if not path.is_file():
            continue
        # Include filenames as well as content, without revealing either on failure.
        text = path.relative_to(ROOT).as_posix() + "\n"
        text += path.read_bytes().decode("utf-8", errors="replace")
        tokens = re.findall(r"[^\W_]+", text.casefold())
        if any(hashlib.sha256(token.encode()).hexdigest() in FORBIDDEN_TOKEN_HASHES
               for token in tokens):
            affected_files += 1

    assert affected_files == 0, f"Retired identifiers found in {affected_files} file(s)"
