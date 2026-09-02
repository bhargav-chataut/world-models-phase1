"""
data_loader.py
Flattens rollout episodes (rollouts_partN.npz, produced by
collect_rollouts.py) into fast, indexable frame data for VAE training.
The VAE trains frame-by-frame -- episode/sequence structure doesn't
matter here, that's what M (the MDN-RNN) uses later.

Two pieces:
  1. flatten_to_parts() - ONE-TIME offline pass. For each
     rollouts_partN.npz, flattens that file's episodes into one
     (n_frames_in_part, 64, 64, 3) array and saves it as
     frames_part{N}.npy. Processes and releases one file at a time.
  2. FrameDataset - a torch Dataset that treats all frames_part*.npy
     files together as one virtual dataset, via a cumulative index.
     Each part file is opened with mmap_mode='r' (lazy -- doesn't load
     into RAM until you actually index into it), so random access
     across the whole dataset stays fast without ever loading
     everything into memory.

Run flatten_to_parts() once in a Colab cell, then use FrameDataset for
every training run after that.
"""

import numpy as np
import glob
import os
import sys
import bisect
import torch
from torch.utils.data import Dataset


def flatten_to_parts(data_dir, out_dir, dtype=np.float32, start_idx=None, end_idx=None):
    """
    Reads each rollouts_partN.npz in data_dir ONE AT A TIME, flattens
    that file's episodes into a single (n_frames, 64, 64, 3) array, and
    saves it as out_dir/frames_part{N}.npy. The source npz's decompressed
    data is released (goes out of scope, garbage collected) before the
    next file is loaded -- this is what keeps peak RAM bounded instead
    of growing across the whole loop.

    Writes go straight to out_dir. If out_dir is a Drive-mounted path,
    each part write is a few hundred MB to 1-2GB, which is a much safer
    chunk size for Drive's network filesystem than either (a) thousands
    of tiny per-episode writes or (b) one giant multi-GB write.

    start_idx / end_idx (optional): process only part_files[start_idx:end_idx]
    instead of all of them. Lets you split a big run across multiple cells
    or sessions, e.g. for 20 total part files:

        flatten_to_parts(data_dir, out_dir, start_idx=0, end_idx=10)
        # ... run again later, same out_dir, resumes with the next 10 ...
        flatten_to_parts(data_dir, out_dir, start_idx=10, end_idx=20)

    The manifest is additive across calls -- entries for parts already
    recorded are kept, and newly processed parts are appended, so you can
    call this as many times as you like with different ranges and end up
    with one complete manifest.npy covering everything processed so far.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_part_files = sorted(glob.glob(os.path.join(data_dir, "rollouts_part*.npz")))
    assert len(all_part_files) > 0, f"No rollout files found in {data_dir}"

    part_files = all_part_files[start_idx:end_idx]
    # Keep track of each file's original index so output filenames stay
    # stable and unique across multiple partial calls (e.g. calling with
    # start_idx=10 doesn't collide with the frames_part0..9.npy from an
    # earlier call).
    indices = list(range(len(all_part_files)))[start_idx:end_idx]

    print(f"Found {len(all_part_files)} part files total. "
          f"Processing {len(part_files)} of them "
          f"(indices {indices[0]}-{indices[-1]})...")
    sys.stdout.flush()

    # Load existing manifest if this out_dir already has one from a
    # previous partial call, so we can append to it rather than overwrite.
    manifest_path = os.path.join(out_dir, "manifest.npy")
    if os.path.exists(manifest_path):
        manifest = [tuple(row) for row in np.load(manifest_path, allow_pickle=True)]
        already_done = {name for name, _ in manifest}
        print(f"Found existing manifest with {len(manifest)} parts already done.")
    else:
        manifest = []
        already_done = set()

    for i, pf in zip(indices, part_files):
        out_name = f"frames_part{i}.npy"
        if out_name in already_done:
            print(f"  [{i}] {out_name} already exists, skipping.")
            continue

        size_mb = os.path.getsize(pf) / 1e6
        print(f"  [{i}] loading {os.path.basename(pf)} "
              f"({size_mb:.1f} MB)...", end=" ")
        sys.stdout.flush()

        with np.load(pf, allow_pickle=True) as d:
            episodes = [np.asarray(ep, dtype=dtype) for ep in d["frames"]]

        n_frames = sum(len(ep) for ep in episodes)
        print(f"{len(episodes)} episodes, {n_frames} frames. Writing...", end=" ")
        sys.stdout.flush()

        out_path = os.path.join(out_dir, out_name)
        arr = np.empty((n_frames, 64, 64, 3), dtype=dtype)
        idx = 0
        for ep in episodes:
            arr[idx:idx + len(ep)] = ep
            idx += len(ep)
        np.save(out_path, arr)

        manifest.append((out_name, n_frames))
        print(f"saved {out_name}")
        sys.stdout.flush()

        # Explicitly drop references before the next iteration so this
        # file's data doesn't linger in memory into the next loop pass.
        del episodes, arr

        # Save the manifest after every part, not just at the end -- if
        # this cell is interrupted partway, progress so far isn't lost.
        np.save(manifest_path, np.array(manifest, dtype=object))

    total_frames = sum(n for _, n in manifest)
    print(f"Done with this call. Manifest now covers {len(manifest)} parts, "
          f"{total_frames} total frames. Saved to {manifest_path}")
    return out_dir


class FrameDataset(Dataset):
    """Treats all frames_part*.npy files in a directory (produced by
    flatten_to_parts) as one virtual flat dataset. Each part is opened
    with mmap_mode='r', so nothing is loaded into RAM until you actually
    index into a specific frame -- safe for large total dataset sizes."""

    def __init__(self, parts_dir):
        manifest_path = os.path.join(parts_dir, "manifest.npy")
        assert os.path.exists(manifest_path), (
            f"No manifest.npy found in {parts_dir} -- run flatten_to_parts() first."
        )
        manifest = np.load(manifest_path, allow_pickle=True)

        self.arrays = []       # lazy memmaps, one per part file
        self.cum_lengths = []  # cumulative frame count, for index -> (part, offset)
        total = 0
        for fname, n_frames in manifest:
            path = os.path.join(parts_dir, fname)
            self.arrays.append(np.load(path, mmap_mode="r"))
            total += int(n_frames)
            self.cum_lengths.append(total)

        self.total_frames = total
        print(f"FrameDataset: {self.total_frames} frames across "
              f"{len(self.arrays)} part files")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, i):
        # Find which part file this global index falls into via binary
        # search over cumulative lengths, then compute the local offset
        # within that part.
        part_idx = bisect.bisect_right(self.cum_lengths, i)
        prev_cum = self.cum_lengths[part_idx - 1] if part_idx > 0 else 0
        local_idx = i - prev_cum

        frame = np.array(self.arrays[part_idx][local_idx])  # copy out of the mmap page
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # HWC -> CHW
        return frame


if __name__ == "__main__":
    DATA_DIR = "./data"             
    OUT_DIR = "./frames_parts"       
    START_IDX = 0                   
    END_IDX = None                 
 
    flatten_to_parts(
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        start_idx=START_IDX,
        end_idx=END_IDX,
    )
    ds = FrameDataset(OUT_DIR)
    print(f"Sanity check: {len(ds)} frames total, "
          f"first frame shape = {tuple(ds[0].shape)}")
