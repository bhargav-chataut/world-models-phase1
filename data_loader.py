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
import torch
from torch.utils.data import Dataset


def flatten_to_memmap(data_dir, out_path, dtype=np.float32):
    """
    Reads all rollouts_partN.npz files in data_dir, concatenates every
    frame from every episode into one big array, and writes it to
    out_path as a memory-mapped .npy file.

    This costs disk space (num_frames * 64*64*3*4 bytes for float32 —
    roughly 3-4GB for ~1000 CarRacing episodes at ~1000 steps each,
    less if episodes terminate early) but makes training data loading
    fast and simple: no per-file reopening, no object-array overhead.
    """
    part_files = sorted(glob.glob(os.path.join(data_dir, "rollouts_part*.npz")))
    assert len(part_files) > 0, f"No rollout files found in {data_dir}"

    # First pass: count total frames so we can preallocate the memmap
    # at the right size (memmap needs a fixed shape up front).
    total_frames = 0
    for pf in part_files:
        with np.load(pf, allow_pickle=True) as d:
            for ep_frames in d["frames"]:
                total_frames += len(ep_frames)

    print(f"Found {total_frames} total frames across {len(part_files)} files")
    print(f"Allocating memmap at {out_path} "
          f"({total_frames * 64 * 64 * 3 * 4 / 1e9:.2f} GB)")

    mmap = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(total_frames, 64, 64, 3)
    )

    # Second pass: write every frame into the memmap in order
    write_idx = 0
    for pf in part_files:
        with np.load(pf, allow_pickle=True) as d:
            frames = d["frames"]
            for ep_frames in frames:
                n = len(ep_frames)
                mmap[write_idx:write_idx + n] = np.stack(ep_frames).astype(dtype)
                write_idx += n
        print(f"  processed {pf} -> {write_idx}/{total_frames} frames written")

    mmap.flush()
    print(f"Done. Wrote {write_idx} frames to {out_path}")
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