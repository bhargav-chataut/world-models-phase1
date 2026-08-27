"""
data_loader.py
Flattens rollout episodes (rollouts_partN.npz, produced by
collect_rollouts.py) into a single indexable dataset of individual
64x64x3 frames. The VAE trains frame-by-frame — episode/sequence
structure doesn't matter here, that's what M (the MDN-RNN) uses later.

Two pieces:
  1. flatten_to_memmap() - ONE-TIME offline pass. Reads every
     rollouts_partN.npz, writes every frame into a single memory-mapped
     .npy file. Do this once per dataset version; re-run only if you
     collect more rollouts.
  2. FrameDataset - a torch Dataset that reads from that memmap. Random
     access is instant since it's one flat array on disk, not N separate
     npz files you'd otherwise have to reopen on every cache miss.

Run flatten_to_memmap() once in a Colab cell, then use FrameDataset for
every training run after that.
"""

import numpy as np
import glob
import os
import sys
import shutil
import torch
from torch.utils.data import Dataset


def flatten_to_memmap(data_dir, out_path, dtype=np.float32,
                       local_staging="/content/frames_tmp.npy"):
    """
    Reads all rollouts_partN.npz files in data_dir, concatenates every
    frame from every episode into one big array, and writes it to
    out_path as a memory-mapped .npy file.

    Two things this version does differently from a naive implementation,
    both for speed on Colab:

    1. Single decompression pass. Each .npz is opened and decompressed
       exactly once (episode arrays are kept in memory just long enough
       to know total frame count and copy them into the memmap), instead
       of opening every file twice -- once to count, once to copy.

    2. Writes to LOCAL disk first (/content, fast local SSD), then does
       one bulk copy to `out_path` on Drive at the end. Writing thousands
       of small memmap slices directly to a Drive-mounted path (which is
       a network filesystem under the hood) is much slower than writing
       locally and copying once.

    Disk cost: num_frames * 64*64*3*4 bytes for float32 -- roughly
    3-4GB for ~1000 CarRacing episodes at ~1000 steps each, less if
    episodes terminate early (which is common with a random policy).
    """
    part_files = sorted(glob.glob(os.path.join(data_dir, "rollouts_part*.npz")))
    assert len(part_files) > 0, f"No rollout files found in {data_dir}"

    # Single pass: decompress each file once, hold episode frame arrays
    # in a list so we know the total count before allocating the memmap.
    # Plain print+flush instead of tqdm -- tqdm.notebook needs ipywidgets
    # enabled in the frontend and fails SILENTLY (no bar, no error) if
    # it isn't, which is almost certainly what you're hitting.
    print(f"Found {len(part_files)} part files. Starting decompression...")
    sys.stdout.flush()
    all_frames = []
    for i, pf in enumerate(part_files):
        size_mb = os.path.getsize(pf) / 1e6
        print(f"  [{i+1}/{len(part_files)}] loading {os.path.basename(pf)} "
              f"({size_mb:.1f} MB)...", end=" ")
        sys.stdout.flush()
        with np.load(pf, allow_pickle=True) as d:
            n_before = len(all_frames)
            for ep_frames in d["frames"]:
                all_frames.append(np.asarray(ep_frames, dtype=dtype))
        print(f"got {len(all_frames) - n_before} episodes "
              f"(running total: {len(all_frames)})")
        sys.stdout.flush()

    total_frames = sum(len(f) for f in all_frames)
    size_gb = total_frames * 64 * 64 * 3 * 4 / 1e9
    print(f"Total frames: {total_frames} ({size_gb:.2f} GB) -- "
          f"writing to local staging file: {local_staging}")

    # Write to LOCAL /content disk first -- much faster than Drive.
    mmap = np.lib.format.open_memmap(
        local_staging, mode="w+", dtype=dtype, shape=(total_frames, 64, 64, 3)
    )
    write_idx = 0
    report_every = max(1, len(all_frames) // 20)  # ~20 progress lines total
    for i, ep in enumerate(all_frames):
        n = len(ep)
        mmap[write_idx:write_idx + n] = ep
        write_idx += n
        if i % report_every == 0 or i == len(all_frames) - 1:
            pct = 100 * (i + 1) / len(all_frames)
            print(f"  writing episode {i+1}/{len(all_frames)} "
                  f"({pct:.0f}%) -- {write_idx}/{total_frames} frames written")
            sys.stdout.flush()
    mmap.flush()
    del mmap  # release the memmap handle before copying the file

    print(f"Local write done. Copying {local_staging} -> {out_path} (Drive)...")
    shutil.copy(local_staging, out_path)
    print(f"Done. {write_idx} frames available at {out_path}")
    return out_path


class FrameDataset(Dataset):
    """Reads individual frames from the flattened memmap produced by
    flatten_to_memmap(). Fast random access -> safe to use shuffle=True
    in your DataLoader without the thrashing you'd get reopening npz
    files per-sample."""

    def __init__(self, memmap_path):
        # mmap_mode='r' means this does NOT load the whole array into
        # RAM -- pages are read from disk on demand as you index into it.
        self.data = np.load(memmap_path, mmap_mode="r")
        print(f"FrameDataset: {len(self.data)} frames, shape {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        frame = np.array(self.data[i])  # copy out of the memmap page
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # HWC -> CHW
        return frame


if __name__ == "__main__":
    # Example usage in a Colab cell:
    # from data_loader import flatten_to_memmap, FrameDataset
    # flatten_to_memmap("/content/drive/MyDrive/world-models/rollouts",
    #                    "/content/drive/MyDrive/world-models/frames.npy")
    # ds = FrameDataset("/content/drive/MyDrive/world-models/frames.npy")
    pass