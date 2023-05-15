#! /bin/bash
conda activate dev

for file_path in $(ls -1 ./tutorials/*ipynb) ; do
    file=$(basename $file_path)
    jupyter nbconvert --to markdown --output-dir ./docs_src/ --output ${file%.ipynb}.md $file_path
done

mkdocs build