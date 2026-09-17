"""Convert a percent-format Python script into a Jupyter notebook.

`notebook_src.py` is the source of truth: it is an ordinary script, so it can be run and
debugged with `python notebook_src.py` instead of being written blind into notebook cells.
This turns it into the `.ipynb` that gets opened in Colab.

Cell markers:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (subsequent `# ` comment lines become the source)

Usage:
    python tools/py2ipynb.py notebook_src.py session12_zero_simulator.ipynb
"""
import json
import sys


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def demote_markdown(body):
    """Strip the leading '# ' from each line of a markdown cell."""
    out = []
    for line in body.split("\n"):
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def as_source(text):
    """Jupyter stores source as a list of lines, each keeping its trailing newline."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build(cells):
    out = []
    for kind, body in cells:
        if kind == "markdown":
            out.append({"cell_type": "markdown", "metadata": {},
                        "source": as_source(demote_markdown(body))})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": as_source(body)})
    return {
        "cells": out,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "notebook_src.py"
    dst = sys.argv[2] if len(sys.argv) > 2 else "session12_zero_simulator.ipynb"
    cells = parse_cells(open(src, encoding="utf-8").read())
    nb = build(cells)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_md = sum(1 for k, _ in cells if k == "markdown")
    print(f"{src} -> {dst}: {len(cells)} cells ({n_md} markdown, {len(cells)-n_md} code)")


if __name__ == "__main__":
    main()
