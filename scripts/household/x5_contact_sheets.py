"""Labeled contact sheets of the delivered renders (no scene edits)."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO/'data/household/x5_initial'
TITLES = {r['id']: r['name'] for r in json.loads((REPO/'src/roboclock/envs/household/tasks.json').read_text())['tasks']}
ids = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
except OSError:
    font = ImageFont.load_default()
for kind in ['overview', 'agent_view']:
    W, Hh, top = 512, 384, 34
    sheet = Image.new('RGB', (W*5, (Hh+top)*2), (22, 22, 22))
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(ids):
        x, y = (i % 5)*W, (i//5)*(Hh+top)
        sheet.paste(Image.open(OUT/t/f'{kind}.png').convert('RGB').resize((W, Hh)), (x, y+top))
        title = TITLES[t]
        d.text((x+10, y+7), (f'{t}  {title}' if len(title) <= 36 else f'{t}  {title[:35]}…'), font=font, fill=(240, 240, 240))
    sheet.save(OUT/f'{kind}_all.jpg', quality=90)
print('SHEETS')
