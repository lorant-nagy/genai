from pathlib import Path

def flatten_repo_to_text(
    repo_dir: str | Path,
    out_file: str | Path = "repo_flattened.txt",
    exts: tuple[str, ...] = (".py", ".sh"),
    recursive: bool = True,
) -> Path:
    """
    Concatenate all code files with given extensions into a single text file.
    
    Each file is delimited by:
        # <relative/path/from/repo_root>
        ...file contents...
        # end <relative/path/from/repo_root>

    Args:
        repo_dir: Path to the repository root directory.
        out_file: Path to the output text file (can be inside or outside repo_dir).
        exts: Tuple of file extensions to include (case-insensitive).
        recursive: If True, search subdirectories recursively.

    Returns:
        Path to the written output file.
    """
    repo_dir = Path(repo_dir).resolve()
    out_file = Path(out_file).resolve()
    exts = tuple(ext.lower() for ext in exts)

    skip_dirs = {
        ".git", ".hg", ".svn",
        "__pycache__", ".mypy_cache", ".pytest_cache",
        "node_modules", ".venv", "venv", "env", ".tox",
        "build", "dist", ".egg-info",
    }

    if recursive:
        candidates = (p for p in repo_dir.rglob("*") if p.is_file())
    else:
        candidates = (p for p in repo_dir.glob("*") if p.is_file())

    files = []
    for p in candidates:
        try:
            if out_file == p.resolve():
                continue
        except Exception:
            pass

        if any(part in skip_dirs for part in p.parts):
            continue

        if p.suffix.lower() in exts:
            files.append(p)

    files.sort(key=lambda p: p.relative_to(repo_dir).as_posix())
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="\n") as out:
        for idx, p in enumerate(files, start=1):
            rel = p.relative_to(repo_dir).as_posix()
            out.write(f"#{rel}\n\n")
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                # Last resort: binary-safe read then decode replace
                data = p.read_bytes()
                text = data.decode("utf-8", errors="replace")
            out.write(text.rstrip() + "\n")
            out.write(f"\n# end {rel}\n")
            if idx < len(files):
                out.write("\n")  # extra blank line between files

    return out_file

if __name__ == "__main__":
    # Example usage:
    # flatten_repo_to_text("/path/to/repo", "all_code.txt")
    # or just:
    # flatten_repo_to_text(".", "all_code.txt")
    out = flatten_repo_to_text(".", "all_code.txt")
    print(f"Wrote: {out}")
