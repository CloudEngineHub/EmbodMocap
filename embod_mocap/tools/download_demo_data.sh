#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hf_base_url="https://huggingface.co/datasets/WenjiaWang/EmbodMocap_release/resolve/main"
cd "$repo_root"

mkdir -p datasets
wget -c -O datasets/dataset_demo.tar \
  "$hf_base_url/dataset_demo.tar"
wget -c -O datasets/release_demo.xlsx \
  "$hf_base_url/release_demo.xlsx"

tar -xf datasets/dataset_demo.tar -C datasets/

echo "Demo dataset is ready in $repo_root/datasets/dataset_demo"
echo "Manifest is ready at $repo_root/datasets/release_demo.xlsx"
