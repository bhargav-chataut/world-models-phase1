"""
collect_rollouts.py

Why random policy: the original paper found random rollouts give the VAE
(V) a diverse enough set of frames to learn a good compressed
representation, without needing a trained agent yet — you can't have a
trained agent before you have V and M, so this has to come first,
chicken-and-egg solved by starting dumb. Random policy would cover more of the 
possible state space than a trained policy would, which is good for VAE training.

Usage:
    python collect_rollouts.py --episodes 1000 --out-dir data --batch-size 50
"""

import argparse
import os
import cv2
import gymnasium as gym
import numpy as np
from tqdm import tqdm


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    return (frame / 255.0).astype(np.float32)


def collect(num_episodes: int, max_steps: int, out_dir: str, batch_size: int = 50) -> None:
    """
    Memory-bounded rollout collector: writes batches of episodes to separate
    compressed .npz files to prevent out-of-memory errors on large episode counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")

    batch_frames = []
    batch_actions = []
    part_idx = 0

    for ep in tqdm(range(num_episodes), desc="Collecting rollouts"):
        obs, _ = env.reset()
        frames, actions = [], []

        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(preprocess_frame(obs))
            actions.append(action)
            if terminated or truncated:
                break

        batch_frames.append(np.array(frames, dtype=np.float32))
        batch_actions.append(np.array(actions, dtype=np.float32))

        # Flush batch to file and release memory
        if (ep + 1) % batch_size == 0 or (ep + 1) == num_episodes:
            part_path = os.path.join(out_dir, f"rollouts_part{part_idx}.npz")
            np.savez_compressed(
                part_path,
                frames=np.array(batch_frames, dtype=object),
                actions=np.array(batch_actions, dtype=object),
            )
            tqdm.write(f"  [Batch {part_idx}] Saved {len(batch_frames)} episodes to {part_path}")
            batch_frames = []
            batch_actions = []
            part_idx += 1

    env.close()
    print(f"\nDone. {num_episodes} episodes saved across {part_idx} files in '{out_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect rollout data using a random policy for World Models.")
    parser.add_argument("--episodes", type=int, default=1000, help="Total number of episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--batch-size", type=int, default=50, help="Episodes saved per .npz chunk.")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory where part files are saved.")
    
    args = parser.parse_args()

    collect(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )