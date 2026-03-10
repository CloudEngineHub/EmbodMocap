#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hf_base_url="https://huggingface.co/datasets/WenjiaWang/EmbodMocap_release/resolve/main"
cd "$repo_root"

mkdir -p checkpoints
wget -c -O checkpoints/ckpt_lazypack.tar \
  "$hf_base_url/ckpt_lazypack.tar"

tar -xf checkpoints/ckpt_lazypack.tar -C checkpoints/
rm -rf checkpoints/ckpt_lazypack_bak
if [ -d checkpoints/ckpt_lazypack ]; then
  mv checkpoints/ckpt_lazypack checkpoints/ckpt_lazypack_bak
  mv checkpoints/ckpt_lazypack_bak/* checkpoints/
  rmdir checkpoints/ckpt_lazypack_bak
fi

echo "Checkpoints are ready in $repo_root/checkpoints"
