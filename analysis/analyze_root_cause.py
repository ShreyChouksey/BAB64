"""
Root cause analysis: WHY is CVP distance always 0?
"""
import hashlib
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, LatticeEngine, CVPSolver

config = BAB256Config(num_basis_vectors=16, coefficient_bound=1)
renderer = BabelRenderer(config)
lattice = LatticeEngine(config)

seed = hashlib.sha256(b"root_cause").digest()
lattice.generate_basis(seed)

target_seed = hashlib.sha256(seed + (0).to_bytes(4, 'big')).digest()
target = renderer.render(target_seed)

print("=== ROOT CAUSE ANALYSIS ===\n")

# 1. What do basis vectors look like?
print("Basis vector stats:")
for i in range(3):
    b = lattice.basis[i].astype(np.float64)
    print(f"  b[{i}]: mean={b.mean():.1f}, std={b.std():.1f}, "
          f"min={b.min()}, max={b.max()}, norm={np.linalg.norm(b):.1f}")

print(f"\nTarget: mean={target.mean():.1f}, std={target.std():.1f}, "
      f"min={target.min()}, max={target.max()}, norm={np.linalg.norm(target):.1f}")

# 2. How similar are basis vectors to each other and to target?
print("\nCosine similarities:")
for i in range(3):
    bi = lattice.basis[i].astype(np.float64)
    cos = np.dot(bi, target.astype(np.float64)) / (np.linalg.norm(bi) * np.linalg.norm(target))
    print(f"  cos(b[{i}], target) = {cos:.6f}")

# 3. The KEY insight - decompose into mean + residual
print("\n=== MEAN/RESIDUAL DECOMPOSITION ===")
mean_vec = np.ones(config.dimension) * 127.5
print(f"  ||mean_vec|| = {np.linalg.norm(mean_vec):.1f}")
print(f"  ||target||   = {np.linalg.norm(target.astype(np.float64)):.1f}")

target_centered = target.astype(np.float64) - mean_vec
print(f"  ||target - mean|| = {np.linalg.norm(target_centered):.1f}")

basis_centered = lattice.basis.astype(np.float64) - mean_vec
print(f"  ||b[0] - mean||  = {np.linalg.norm(basis_centered[0]):.1f}")

# 4. Cosine similarity of centered vectors
print("\nCentered cosine similarities:")
for i in range(3):
    bc = basis_centered[i]
    cos = np.dot(bc, target_centered) / (np.linalg.norm(bc) * np.linalg.norm(target_centered))
    print(f"  cos(b[{i}]-mean, target-mean) = {cos:.6f}")

# 5. What lstsq gives for original vs centered
print("\n=== LSTSQ COEFFICIENTS ===")
B = lattice.basis.astype(np.float64)
t = target.astype(np.float64)
coeffs_orig, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
print(f"Original: coeffs = {coeffs_orig.round(4)}")
print(f"  sum(coeffs) = {coeffs_orig.sum():.4f}")
print(f"  max|c| = {np.max(np.abs(coeffs_orig)):.4f}")

# With centered vectors
Bc = basis_centered
tc = target_centered
coeffs_cent, _, _, _ = np.linalg.lstsq(Bc.T, tc, rcond=None)
print(f"\nCentered: coeffs = {coeffs_cent.round(4)}")
print(f"  sum(coeffs) = {coeffs_cent.sum():.4f}")
print(f"  max|c| = {np.max(np.abs(coeffs_cent)):.4f}")

# Distance with centered Babai
int_coeffs_cent = np.round(coeffs_cent).astype(np.int32)
int_coeffs_cent = np.clip(int_coeffs_cent, -1, 1)
lattice_point = np.dot(int_coeffs_cent.astype(np.int64), basis_centered.astype(np.int64))
diff = lattice_point - target_centered.astype(np.int64)
dist = float(np.sqrt(np.sum(diff ** 2)))
print(f"\nCentered CVP distance (bound=1): {dist:.1f}")
print(f"  (vs target norm {np.linalg.norm(target_centered):.1f})")
