import argparse
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def extract_parquet(parquet_file: Path, output_dir: Path):
    """Extract all files from a parquet archive."""
    df = pd.read_parquet(parquet_file)
    for _, row in df.iterrows():
        out_path = output_dir / row["relative_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(row["content"])


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(
        description="Extract selected LayeredFlow-Syn files."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing parquet files (e.g. data/0)",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output root directory",
    )
    args = parser.parse_args()

    #
    folder_name = args.data_dir.name  # e.g. "0"
    dst_dir = args.output_root / folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Process only non-RGB parquet files
    flow_files = sorted(
        p for p in args.data_dir.glob("*.parquet") if not p.stem.endswith("_rgb")
    )
    for flow_parquet in tqdm(
        flow_files,
        desc=f"Folder {folder_name}",
        unit="pair",
    ):
        rgb_parquet = flow_parquet.with_name(flow_parquet.stem + "_rgb.parquet")

        if not rgb_parquet.exists():
            tqdm.write(f"Skipping {flow_parquet.name}: RGB parquet not found.")
            continue

        # Extract sample index (e.g. 0_100 -> 0100)
        sample_id = int(flow_parquet.stem.split("_")[1])
        prefix = f"{sample_id:04d}"

        # Temporary extraction directory
        tmpdir = Path(tempfile.mkdtemp())
        flow_extract = tmpdir / "flow"
        rgb_extract = tmpdir / "rgb"
        extract_parquet(flow_parquet, flow_extract)
        extract_parquet(rgb_parquet, rgb_extract)
        files = [
            # Left camera
            (
                rgb_extract / "frame0_l.png",
                dst_dir / f"{prefix}_frame0_l.png",
            ),
            (
                rgb_extract / "frame1_l.png",
                dst_dir / f"{prefix}_frame1_l.png",
            ),
            (
                flow_extract / "frame0" / "left" / "flow_layer00_forward.flo",
                dst_dir / f"{prefix}_flow_layer00_forward_l.flo",
            ),
            (
                flow_extract / "frame0" / "left" / "mask_layer00.png",
                dst_dir / f"{prefix}_mask_layer00_forward_l.png",
            ),
            (
                flow_extract / "frame0" / "left" / "mask_layer01.png",
                dst_dir / f"{prefix}_mask_layer01_forward_l.png",
            ),
            (
                flow_extract / "frame0" / "left" / "mask_layer02.png",
                dst_dir / f"{prefix}_mask_layer02_forward_l.png",
            ),
            (
                flow_extract / "frame0" / "left" / "mask_layer03.png",
                dst_dir / f"{prefix}_mask_layer03_forward_l.png",
            ),
            (
                flow_extract / "frame1" / "left" / "flow_layer00_backward.flo",
                dst_dir / f"{prefix}_flow_layer00_backward_l.flo",
            ),
            (
                flow_extract / "frame1" / "left" / "mask_layer00.png",
                dst_dir / f"{prefix}_mask_layer00_backward_l.png",
            ),
            (
                flow_extract / "frame1" / "left" / "mask_layer01.png",
                dst_dir / f"{prefix}_mask_layer01_backward_l.png",
            ),
            (
                flow_extract / "frame1" / "left" / "mask_layer02.png",
                dst_dir / f"{prefix}_mask_layer02_backward_l.png",
            ),
            (
                flow_extract / "frame1" / "left" / "mask_layer03.png",
                dst_dir / f"{prefix}_mask_layer03_backward_l.png",
            ),
            # Right camera
            (
                rgb_extract / "frame0_r.png",
                dst_dir / f"{prefix}_frame0_r.png",
            ),
            (
                rgb_extract / "frame1_r.png",
                dst_dir / f"{prefix}_frame1_r.png",
            ),
            (
                flow_extract / "frame0" / "right" / "flow_layer00_forward.flo",
                dst_dir / f"{prefix}_flow_layer00_forward_r.flo",
            ),
            (
                flow_extract / "frame0" / "right" / "mask_layer00.png",
                dst_dir / f"{prefix}_mask_layer00_forward_r.png",
            ),
            (
                flow_extract / "frame0" / "right" / "mask_layer01.png",
                dst_dir / f"{prefix}_mask_layer01_forward_r.png",
            ),
            (
                flow_extract / "frame0" / "right" / "mask_layer02.png",
                dst_dir / f"{prefix}_mask_layer02_forward_r.png",
            ),
            (
                flow_extract / "frame0" / "right" / "mask_layer03.png",
                dst_dir / f"{prefix}_mask_layer03_forward_r.png",
            ),
            (
                flow_extract / "frame1" / "right" / "flow_layer00_backward.flo",
                dst_dir / f"{prefix}_flow_layer00_backward_r.flo",
            ),
            (
                flow_extract / "frame1" / "right" / "mask_layer00.png",
                dst_dir / f"{prefix}_mask_layer00_backward_r.png",
            ),
            (
                flow_extract / "frame1" / "right" / "mask_layer01.png",
                dst_dir / f"{prefix}_mask_layer01_backward_r.png",
            ),
            (
                flow_extract / "frame1" / "right" / "mask_layer02.png",
                dst_dir / f"{prefix}_mask_layer02_backward_r.png",
            ),
            (
                flow_extract / "frame1" / "right" / "mask_layer03.png",
                dst_dir / f"{prefix}_mask_layer03_backward_r.png",
            ),
        ]

        #
        for src, dst in files:
            if src.exists():
                shutil.copy2(src, dst)
            else:
                tqdm.write(f"Missing: {src}")
        shutil.rmtree(tmpdir)
    print(f"Done. Output saved to {dst_dir}")
