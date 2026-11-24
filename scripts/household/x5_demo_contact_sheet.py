"""Contact sheet of one recorded Dual-X5 episode: six instants, the overview and the three policy views.

    python scripts/household/x5_demo_contact_sheet.py RUN_DIR

Reads RUN_DIR/<TASK>.mp4, head.mp4, left_wrist.mp4 and right_wrist.mp4 (frame-synchronized) and
frames.json; writes RUN_DIR/contact_sheet.jpg. Rows: overview, head camera (rgb), left | right wrist.
Needs ffmpeg on PATH.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLS, CW, CH = 6, 400, 300


def frames(path, indices, size):
    """Decode the given frame indices of a video at size (w, h)."""
    select = '+'.join(f'eq(n\\,{i})' for i in indices)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f"select='{select}',scale={size[0]}:{size[1]}",
                          '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0], 3)


def main(run):
    run = Path(run)
    result = json.loads((run/'result.json').read_text())
    task = result['task']
    times = json.loads((run/'frames.json').read_text())['sim_time_s']
    picks = np.linspace(0, len(times)-1, COLS).round().astype(int).tolist()
    rows = [frames(run/f'{task}.mp4', picks, (CW, CH)), frames(run/'head.mp4', picks, (CW, CH))]
    left, right = frames(run/'left_wrist.mp4', picks, (CW//2, CH//2)), frames(run/'right_wrist.mp4', picks, (CW//2, CH//2))
    wrist = np.zeros((COLS, CH//2, CW, 3), np.uint8)
    wrist[:, :, :CW//2], wrist[:, :, CW//2:] = left, right
    top, label = 30, 22
    sheet = Image.new('RGB', (COLS*CW, top+2*(CH+label)+CH//2+label), (22, 22, 22))
    d = ImageDraw.Draw(sheet)
    d.text((10, 9), f'{task} | {result["difficulty"]} seed {result["seed"]} | overview, head camera (rgb), left | right wrist',
           fill=(240, 240, 240))
    y = top
    for name, row, h in (('overview', rows[0], CH), ('head', rows[1], CH), ('wrists', wrist, CH//2)):
        for k, img in enumerate(row):
            sheet.paste(Image.fromarray(img), (k*CW, y+label))
            if name == 'overview':
                d.text((k*CW+6, y+5), f't = {times[picks[k]]:.1f} s', fill=(230, 230, 230))
        y += h+label
    sheet.save(run/'contact_sheet.jpg', quality=88)
    print(run/'contact_sheet.jpg')


if __name__ == '__main__':
    main(sys.argv[1])
