#! /bin/bash
# Compile the mkdocs site from notebook tutorials.
#
# Picks up:
#   - flat top-level notebooks at  ./tutorials/*.ipynb
#   - per-dataset notebooks at     ./tutorials/<name>/<name>.ipynb
#
# Use `uv run` so the script works inside the .venv created by setup_uv.sh.
# Fall back to plain `jupyter` if `uv` isn't on PATH (e.g. conda envs).

set -e

if command -v uv >/dev/null 2>&1; then
    NBCONVERT="uv run jupyter nbconvert"
else
    NBCONVERT="jupyter nbconvert"
fi

# `find -maxdepth 2` catches both ./tutorials/foo.ipynb and
# ./tutorials/<name>/<name>.ipynb without descending further.
for file_path in $(find ./tutorials -maxdepth 2 -name '*.ipynb' \
                                    -not -path '*/.ipynb_checkpoints/*') ; do
    file=$(basename "$file_path")
    $NBCONVERT --to markdown \
               --output-dir ./docs_src/ \
               --output "${file%.ipynb}.md" \
               "$file_path"
done

# Only to host locally; the GitHub Action builds the live site automatically.
# mkdocs build