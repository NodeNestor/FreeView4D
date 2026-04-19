"""Convert mp4 to GIF using imageio. Used only for generating README assets."""
import argparse
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()

    reader = imageio.get_reader(args.input)
    frames = []
    for f in reader:
        h, w = f.shape[:2]
        new_w = args.width
        new_h = int(h * new_w / w)
        img = Image.fromarray(f).resize((new_w, new_h), Image.LANCZOS)
        frames.append(np.array(img))
    imageio.mimsave(args.output, frames, fps=args.fps, loop=0)
    print(f"[save] {args.output}  ({len(frames)} frames, {args.width}x{new_h}, {args.fps} fps)")


if __name__ == "__main__":
    main()
