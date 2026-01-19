"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path, max_count_size: int = 50_000_000) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.

    For large files (>max_count_size bytes), estimates line count from file size
    to avoid hanging on huge files.
    """
    file_size = f.stat().st_size

    if f.suffix in plaintext_files:
        # For large files, estimate line count from file size to avoid hanging
        if file_size > max_count_size:
            # Sample first 1k to estimate average line length
            sample_size = min(1000, file_size)
            with open(f, 'rb') as fp:
                sample = fp.read(sample_size)
            sample_lines = sample.count(b'\n')
            if sample_lines > 0:
                avg_line_len = sample_size / sample_lines
                estimated_lines = int(file_size / avg_line_len)
                return estimated_lines, f"~{estimated_lines:,} lines (estimated)"
            else:
                return file_size, humanize.naturalsize(file_size)
        else:
            num_lines = sum(1 for _ in open(f))
            return num_lines, f"{num_lines} lines"
    else:
        return file_size, humanize.naturalsize(file_size)


def file_tree(path: Path, depth=0,max_dirs=20) -> str:
    """Generate a tree structure of files in a directory"""
    result = []
    files = [p for p in Path(path).iterdir() if not p.is_dir()]
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs)[:max_dirs]:
        result.append(f"{' ' * depth * 4}{p.name}/")
        result.append(file_tree(p, depth + 1, max_dirs))

    if len(dirs) > max_dirs:
        result.append(f"{' ' * depth * 4}... and {len(dirs) - max_dirs} other subfolders")

    return "\n".join(result)


def _walk(path: Path):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True, max_rows: int = 1000) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.
        max_rows (int, optional): maximum rows to read for preview. Defaults to 1000.

    Returns:
        str: the textual preview
    """
    # Check file size to determine if we need to sample
    file_size = p.stat().st_size
    is_large_file = file_size > 50_000_000  # 50MB threshold

    if is_large_file:
        # For large files, only read a sample
        df = pd.read_csv(p, nrows=max_rows)
        # Estimate total rows from file size
        estimated_rows, _ = get_file_len_size(p)
        row_info = f"~{estimated_rows:,} rows (sampled {max_rows:,})"
    else:
        df = pd.read_csv(p)
        row_info = f"{df.shape[0]} rows"

    out = []

    out.append(f"-> {file_name} has {row_info} and {df.shape[1]} columns.")

    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str, max_lines: int = 1000):
    """Generate a textual preview of a json file using a generated json schema

    Args:
        p (Path): the path to the json file
        file_name (str): the file name to use in the preview
        max_lines (int, optional): maximum lines to read for JSONL files. Defaults to 1000.
    """
    builder = SchemaBuilder()
    with open(p) as f:
        first_line = f.readline().strip()

        try:
            first_object = json.loads(first_line)

            if not isinstance(first_object, dict):
                raise json.JSONDecodeError("The first line isn't JSON", first_line, 0)

            # if the the next line exists and is not empty, then it is a JSONL file
            second_line = f.readline().strip()
            if second_line:
                f.seek(0)  # so reset and read line by line, but limit to max_lines
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    builder.add_object(json.loads(line.strip()))
            # if it is empty, then it's a single JSON object file
            else:
                builder.add_object(first_object)

        except json.JSONDecodeError:
            # if first line isn't JSON, then it's prettified and we can read whole file
            # but check file size first
            file_size = p.stat().st_size
            if file_size > 50_000_000:  # 50MB threshold
                return f"-> {file_name} is a large JSON file ({humanize.naturalsize(file_size)}), schema preview skipped"
            f.seek(0)
            builder.add_object(json.load(f))

    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
        indent=2
    )


def generate(base_path, include_file_details=True, simple=False):
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files
    """
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    if len(result) > 6_000 and not simple:
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )
    # if still too long, we truncate
    if len(result) > 6_000 and simple:
        return result[:6_000] + "\n... (truncated)"

    return result
