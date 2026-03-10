"""Lightweight SMPL mesh downsampler for visualization."""

from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix


class SMPLMeshSampler:
    """SMPL mesh sampler with level-based downsampling.

    This class is intentionally minimal for visualization usage:
    - loads downsampling matrices and faces from mesh_downsampling.npz
    - supports downsample for (N, 3) and (T, N, 3) numpy arrays
    """

    def __init__(self, npz_path=None):
        if npz_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            npz_path = repo_root / "body_models" / "smpl" / "mesh_downsampling.npz"

        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=True, encoding="latin1")

        self.D = [csc_matrix(data["D"][i]) for i in range(len(data["D"]))]
        self.num_levels = len(self.D) + 1

        self.num_vertices = [self.D[0].shape[1]]
        self.num_vertices.extend([mat.shape[0] for mat in self.D])

        if "F" in data:
            self.faces = [np.asarray(data["F"][i]).astype(np.int64) for i in range(len(data["F"]))]
        else:
            self.faces = None

    def _infer_level_from_vertices(self, vertices):
        nverts = int(vertices.shape[1] if vertices.ndim == 3 else vertices.shape[0])
        for level, count in enumerate(self.num_vertices):
            if nverts == int(count):
                return level
        raise ValueError(f"Unrecognized vertex count {nverts}, supported={self.num_vertices}")

    def downsample(self, vertices, from_level=None, to_level=1):
        """Downsample mesh vertices.

        Args:
            vertices: (N, 3) or (T, N, 3) numpy array
            from_level: source level; inferred from vertex count if None
            to_level: target level
        """
        vertices = np.asarray(vertices)

        if from_level is None:
            from_level = self._infer_level_from_vertices(vertices)

        if not (0 <= from_level < self.num_levels):
            raise ValueError(f"Invalid from_level={from_level}")
        if not (0 <= to_level < self.num_levels):
            raise ValueError(f"Invalid to_level={to_level}")
        if from_level >= to_level:
            return vertices

        is_batched = vertices.ndim == 3
        if not is_batched:
            vertices = vertices[np.newaxis, ...]

        result = vertices
        for level in range(from_level, to_level):
            result = np.stack([self.D[level].dot(frame) for frame in result], axis=0)

        if not is_batched:
            result = result[0]
        return result

    def get_faces(self, level=0):
        if self.faces is None:
            return None
        if not (0 <= level < len(self.faces)):
            raise ValueError(f"Invalid level={level}")
        return self.faces[level]
