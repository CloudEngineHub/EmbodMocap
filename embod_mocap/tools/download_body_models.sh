#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hf_base_url="https://huggingface.co/datasets/WenjiaWang/EmbodMocap_release/resolve/main"
cd "$repo_root"

mkdir -p body_models/smpl
wget -c -O body_models/smpl/SMPL_NEUTRAL.pkl \
  "$hf_base_url/SMPL_NEUTRAL.pkl"
wget -c -O body_models/smpl/J_regressor_extra.npy \
  "$hf_base_url/J_regressor_extra.npy"
wget -c -O body_models/smpl/J_regressor_h36m.npy \
  "$hf_base_url/J_regressor_h36m.npy"
wget -c -O body_models/smpl/mesh_downsampling.npz \
  "$hf_base_url/mesh_downsampling.npz"
wget -c -O body_models/smpl/smpl_mean_params.npz \
  "$hf_base_url/smpl_mean_params.npz"

echo "SMPL assets are ready in $repo_root/body_models/smpl"
