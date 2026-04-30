#!/usr/bin/env python3
"""Render ball-and-stick PNG snapshots for all A3HT NEMD runs."""

import multiprocessing
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from config import get as _cfg, ROOT as _REPO_ROOT  # noqa: E402

_lammps_dir = _cfg("paths.lammps_dir", "lammps-30Mar2026/build-cray-rebo2")
_lammps_dir = _lammps_dir if Path(_lammps_dir).is_absolute() else str(_REPO_ROOT / _lammps_dir)
LMP = str(Path(_lammps_dir).parent / "build-dump-image" / "lmp")
RUNS_ROOT = Path(os.environ.get("A3HT_RUNS_ROOT", str(_REPO_ROOT / "my_runs")))
PYTHON = _cfg("paths.python", sys.executable)
WORKERS = 8
SNAPSHOT_NAME = "snapshot.png"

LAMMPS_TEMPLATE = textwrap.dedent("""\
    units           metal
    atom_style      bond
    boundary        p p p
    read_data       {data_file}
    bond_style      zero nocoeff
    bond_coeff      *
    region          slice block 0 10 INF INF INF INF
    dump            img all image 1 {ppm_file} type type &
                    zoom 1.4 size 800 1600 &
                    view 90 0 &
                    box no 0 &
                    axes no 0 0
    dump_modify     img region slice
    dump_modify     img backcolor white
    dump_modify     img acolor 1 cyan
    dump_modify     img adiam 1 0.7
    dump_modify     img bcolor 1 gray
    dump_modify     img bdiam 1 0.35
    run             0
""")


def atomic_to_bond(src: Path, dst: Path) -> None:
    atoms, box = [], {}
    in_atoms = False
    with open(src) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if 'xlo xhi' in s:
                p = s.split(); box['xlo'], box['xhi'] = float(p[0]), float(p[1])
            elif 'ylo yhi' in s:
                p = s.split(); box['ylo'], box['yhi'] = float(p[0]), float(p[1])
            elif 'zlo zhi' in s:
                p = s.split(); box['zlo'], box['zhi'] = float(p[0]), float(p[1])
            elif s == 'Atoms # atomic':
                in_atoms = True
            elif in_atoms and s and s[0].isdigit():
                p = s.split()
                if len(p) >= 5:
                    atoms.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4])))
            elif in_atoms and s and not s[0].isdigit():
                in_atoms = False

    pos = np.array([[a[2], a[3], a[4]] for a in atoms])
    pairs = list(cKDTree(pos).query_pairs(1.85))

    with open(dst, 'w') as f:
        f.write("LAMMPS bond-style data file\n\n")
        f.write(f"{len(atoms)} atoms\n{len(pairs)} bonds\n\n")
        f.write("1 atom types\n1 bond types\n\n")
        f.write(f"{box['xlo']:.10f} {box['xhi']:.10f} xlo xhi\n")
        f.write(f"{box['ylo']:.10f} {box['yhi']:.10f} ylo yhi\n")
        f.write(f"{box['zlo']:.10f} {box['zhi']:.10f} zlo zhi\n\n")
        f.write("Masses\n\n1 12.011\n\n")
        f.write("Atoms # bond\n\n")
        for a in atoms:
            f.write(f"{a[0]} 1 {a[1]} {a[2]:.10f} {a[3]:.10f} {a[4]:.10f}\n")
        f.write("\nBonds\n\n")
        for k, (i, j) in enumerate(pairs, 1):
            f.write(f"{k} 1 {atoms[i][0]} {atoms[j][0]}\n")


def trim_whitespace(ppm: Path, png: Path, pad: int = 10) -> None:
    img = Image.open(ppm).convert('RGB')
    arr = np.array(img)
    mask = ~((arr[:, :, 0] == 255) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 255))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        img.save(png)
        return
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img.crop((
        max(0, cmin - pad), max(0, rmin - pad),
        min(arr.shape[1], cmax + pad), min(arr.shape[0], rmax + pad)
    ))
    cropped.save(png)


def render_one(nemd_data: Path) -> str:
    run_dir = nemd_data.parent.parent
    out_png = run_dir / SNAPSHOT_NAME

    if out_png.exists():
        return f"SKIP  {run_dir.name}"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bond_data = tmp / "bond.data"
            lmp_in    = tmp / "render.lmp"
            ppm_pat   = tmp / "snap.*.ppm"
            ppm_out   = tmp / "snap.0.ppm"

            atomic_to_bond(nemd_data, bond_data)

            lmp_in.write_text(LAMMPS_TEMPLATE.format(
                data_file=bond_data,
                ppm_file=ppm_pat,
            ))

            result = subprocess.run(
                [LMP, "-in", str(lmp_in)],
                capture_output=True, text=True
            )
            if result.returncode != 0 or not ppm_out.exists():
                return f"ERROR {run_dir.name}: lammps failed\n{result.stderr[-300:]}"

            trim_whitespace(ppm_out, out_png)

        return f"OK    {run_dir.name}  →  {out_png}"

    except Exception as e:
        return f"ERROR {run_dir.name}: {e}"


def main():
    nemd_files = sorted(
        RUNS_ROOT.glob("*/data/gc_rebo2_nemd.data"),
        key=lambda p: int(p.parts[-3])
    )
    print(f"Found {len(nemd_files)} runs with NEMD data. Workers={WORKERS}")

    with multiprocessing.Pool(WORKERS) as pool:
        for msg in pool.imap_unordered(render_one, nemd_files):
            print(msg, flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
