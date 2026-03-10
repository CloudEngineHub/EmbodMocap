#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hf_base_url="https://huggingface.co/datasets/WenjiaWang/EmbodMocap_release/resolve/main"
cd "$repo_root"

mkdir -p datasets
wget -c -O datasets/dataset_release.tar \
  "$hf_base_url/dataset_release.tar"
wget -c -O datasets/release.xlsx \
  "$hf_base_url/release.xlsx"

tar -xf datasets/dataset_release.tar -C datasets/

echo "Full dataset is ready in $repo_root/datasets/dataset_release"
echo "Manifest is ready at $repo_root/datasets/release.xlsx"
