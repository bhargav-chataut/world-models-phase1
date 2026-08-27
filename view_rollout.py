"""
view_rollout.py

Inspect saved rollout chunks: view a single static frame or play the whole episode.

Usage:
    # Play as video:
    python view_rollout.py --file data/rollouts_part0.npz --episode 0 --fps 30

    # View a single frame:
    python view_rollout.py --file data/rollouts_part0.npz --episode 0 --step 0
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def run(file_path: str, episode_idx: int = 0, step_idx: int | None = None, fps: int = 30) -> None:
    data = np.load(file_path, allow_pickle=True)
    frames = data["frames"]
    actions = data["actions"]

    episode_frames = frames[episode_idx]
    episode_actions = actions[episode_idx]
    total_steps = len(episode_frames)

    print(f"Loaded: {file_path}")
    print(f"Total episodes in batch: {len(frames)}")
    print(f"Episode {episode_idx} length: {total_steps} steps")

    # Mode 1: Static single frame inspection
    if step_idx is not None:
        frame = np.asarray(episode_frames[step_idx], dtype=np.float32)
        action = episode_actions[step_idx]
        print(f"Frame shape: {frame.shape} | dtype: {frame.dtype}")
        print(f"Action at step {step_idx}: {action}")

        plt.imshow(frame)
        plt.title(f"File: {file_path} | Ep: {episode_idx} | Step: {step_idx}")
        plt.axis("off")
        plt.show()
        return

    # Mode 2: Play as video
    print(f"Playing Episode {episode_idx} ({total_steps} steps) at ~{fps} FPS...")
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.asarray(episode_frames[0], dtype=np.float32))
    ax.axis("off")
    delay = 1.0 / fps

    for step, (frame, action) in enumerate(zip(episode_frames, episode_actions)):
        frame_arr = np.asarray(frame, dtype=np.float32)
        img_display.set_data(frame_arr)
        ax.set_title(
            f"Ep: {episode_idx} | Step: {step}/{total_steps}\n"
            f"Steer: {action[0]:.2f}, Gas: {action[1]:.2f}, Brake: {action[2]:.2f}"
        )
        plt.pause(delay)
        if not plt.fignum_exists(fig.number):
            break

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or play back saved rollout episodes.")
    parser.add_argument("--file", type=str, default="data/rollouts_part0.npz", help="Path to .npz chunk.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index inside the chunk.")
    parser.add_argument("--step", type=int, default=None, help="Specific step to show statically. Omit to play as video.")
    parser.add_argument("--fps", type=int, default=30, help="Playback speed in frames per second (video mode).")

    args = parser.parse_args()
    run(file_path=args.file, episode_idx=args.episode, step_idx=args.step, fps=args.fps)