#!/usr/bin/env python3
from __future__ import annotations

import ast
import datetime
import subprocess
import sys
from pathlib import Path

# ================== CONFIG ==================
AUTHOR = "Ahmad Tarraf"
INSTITUTION = "TU Darmstadt, Germany"
LICENSE_NAME = "BSD 3-Clause License"
LICENSE_URL = "https://github.com/tuda-parallel/FTIO/blob/main/LICENSE"
# ============================================

CURRENT_YEAR = datetime.datetime.now().year
VERSION = sys.argv[1] if len(sys.argv) > 1 else "0.0.0"
UPDATE_PATH = sys.argv[2] if len(sys.argv) > 2 else None
DEFAULT_ROOT = "./ftio"

SCRIPT_NAME = Path(__file__).name  # skip updating self


# ---------------------------------------------------------------------------
def git_creation_date(path: Path) -> str | None:
    """Return first commit date as 'Mon YYYY' in English, or None."""
    try:
        result = subprocess.run(
            ["git", "log", "--reverse", "--format=%ad", "--date=format:%b %Y", str(path)],
            capture_output=True,
            text=True,
            check=True,
            env={**dict(**subprocess.os.environ), "LC_TIME": "C"},  # force English months
        )
        lines = result.stdout.strip().splitlines()
        return lines[0] if lines else None
    except Exception:
        return None


def git_original_author(path: Path) -> str | None:
    """Return the first committer's name for the file, or None."""
    try:
        result = subprocess.run(
            ["git", "log", "--reverse", "--format=%an", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().splitlines()
        return lines[0] if lines else None
    except Exception:
        return None


def extract_field(docstring: str, field: str) -> str | None:
    for line in docstring.splitlines():
        if line.strip().startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return None


def strip_existing_metadata(docstring: str) -> str:
    """Remove existing metadata so we can rebuild it idempotently."""
    markers = (
        "Author:",
        "Editor:",
        "Copyright",
        "Version:",
        "Date:",
        "Licensed under the",
    )
    lines = docstring.splitlines()
    for i, line in enumerate(lines):
        if any(line.strip().startswith(m) for m in markers):
            return "\n".join(lines[:i]).rstrip()
    return docstring.rstrip()


def build_header_metadata(
    docstring: str | None, git_date: str | None, path: Path | None
) -> str:
    """
    Build module header metadata block.
    - Preserve existing Author if present
    - Else use Git original author if available
    - Else fallback to configured AUTHOR
    - Preserve Editor, Date, Version, License
    """
    lines = []

    # Author
    existing_author = extract_field(docstring, "Author") if docstring else None
    if existing_author:
        lines.append(f"Author: {existing_author}")
    else:
        git_author = git_original_author(path) if path else None
        author_name = git_author if git_author else AUTHOR
        lines.append(f"Author: {author_name}")

    # Editor
    existing_editor = extract_field(docstring, "Editor") if docstring else None
    if existing_editor:
        lines.append(f"Editor: {existing_editor}")

    # Copyright / Version / Date
    if CURRENT_YEAR > 2024:
        lines.append(f"Copyright (c) 2024-{CURRENT_YEAR} {INSTITUTION}")
    else:
        lines.append(f"Copyright (c) {CURRENT_YEAR} {INSTITUTION}")
    lines.append(f"Version: {VERSION}")

    existing_date = extract_field(docstring, "Date") if docstring else None
    if existing_date:
        lines.append(f"Date: {existing_date}")
    elif git_date:
        lines.append(f"Date: {git_date}")

    # License
    lines.extend(
        [
            "",
            f"Licensed under the {LICENSE_NAME}.",
            "For more information, see the LICENSE file in the project root:",
            LICENSE_URL,
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
def update_file(path: Path):
    """Update the module docstring header metadata for a single Python file."""
    if path.name == SCRIPT_NAME:
        return

    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"Skipping invalid Python file: {path}")
        return

    docstring = ast.get_docstring(tree, clean=False)
    git_date = git_creation_date(path)

    if docstring:
        clean_text = strip_existing_metadata(docstring)
        metadata = build_header_metadata(docstring, git_date, path)

        new_docstring = f"{clean_text}\n\n{metadata}".strip()

        doc_node = tree.body[0]
        if not isinstance(doc_node, ast.Expr) or not isinstance(
            doc_node.value, ast.Constant
        ):
            return

        old_literal = ast.get_source_segment(source, doc_node)
        quote = old_literal[:3]  # """ or '''
        replacement = f"{quote}\n{new_docstring}\n{quote}\n\n"
        updated_source = source.replace(old_literal, replacement, 1)
    else:
        metadata = build_header_metadata(None, git_date, path)
        updated_source = f'"""\n{metadata}\n"""\n\n{source}'

    path.write_text(updated_source, encoding="utf-8")


# ---------------------------------------------------------------------------
def update_project(root: str):
    """Recursively update Python files under a directory or a single file."""
    path = Path(root)
    files_to_update = []

    if path.is_file() and path.suffix == ".py":
        files_to_update.append(path)
    elif path.is_dir():
        for py_file in path.rglob("*.py"):
            files_to_update.append(py_file)

    # Skip the updater script itself
    files_to_update = [f for f in files_to_update if f.name != SCRIPT_NAME]

    # Print summary
    if files_to_update:
        dirs = {f.parent for f in files_to_update}
        print(f"Updating {len(files_to_update)} Python files in {len(dirs)} directories:")
        for d in sorted(dirs):
            print(f"  {d}")
    else:
        print("No Python files found to update.")
        return

    # Update files
    for py_file in files_to_update:
        update_file(py_file)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if UPDATE_PATH:
        update_project(UPDATE_PATH)
    else:
        update_project(DEFAULT_ROOT)
