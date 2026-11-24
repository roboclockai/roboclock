"""Deterministic authored textures for initial-state cues (seeded procedural images, no photographs).

Every texture here is a disclosed appearance proxy. Process-dependent cues
(pancake bubbles, cheese browning, toast char, smoke density, scale reading)
show only the initial state; the simulation stage must drive them from its
process model instead of swapping images on a timer.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]/'assets/x5/textures'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
MONO_B = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def noise(shape, scale, octaves=4):
    out = np.zeros(shape)
    for o in range(octaves):
        s = max(2, int(scale/(2**o)))
        small = rng.normal(0, 1, (shape[0]//s+2, shape[1]//s+2))
        img = Image.fromarray(small.astype(np.float32), mode='F').resize((shape[1], shape[0]), Image.BICUBIC)
        out += np.asarray(img)/(2**o)
    return out/np.abs(out).max()


def save(name, arr):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(OUT/name)


N = 512
y, x = np.mgrid[0:N, 0:N]/N
r = np.hypot(x-.5, y-.5)

# Pancake top during first-side cooking: matte batter with forming and a few popped bubbles.
base = np.array([226, 196, 132.])
tex = base+18*noise((N, N), 64)[..., None]*np.array([1, .9, .6])
img = Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))
d = ImageDraw.Draw(img)
for i in range(70):
    cx, cy = rng.uniform(.12, .88, 2)*N
    if np.hypot(cx/N-.5, cy/N-.5) > .40:
        continue
    rad = rng.uniform(3, 8)
    popped = rng.random() < .35
    d.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(150, 112, 62) if popped else (238, 214, 160),
              outline=(185, 150, 92))
edge = np.clip((r-.40)/.08, 0, 1)[..., None]
arr = np.asarray(img).astype(float)*(1-edge*.25)+edge*np.array([196, 150, 80])*.25
save('pancake_top_first_side.png', arr)

# Cheese toast (initial: cheese just melting, pale, no browning yet).
bread = np.array([214, 172, 104.])+22*noise((N, N), 48)[..., None]
crust = np.clip((np.maximum(np.abs(x-.5), np.abs(y-.5))-.43)/.05, 0, 1)[..., None]
bread = bread*(1-crust)+crust*np.array([142, 88, 40.])
cheese_mask = (np.maximum(np.abs(x-.5)*1.05, np.abs(y-.5))+.03*noise((N, N), 40)) < .38
cheese = np.array([246, 214, 110.])+14*noise((N, N), 24)[..., None]
arr = np.where(cheese_mask[..., None], cheese, bread)
save('cheese_toast_initial.png', arr)

# French toast slice: bread soaked in egg mixture (raw, pale yellow, wet sheen spots).
egg = np.array([232, 200, 118.])+20*noise((N, N), 40)[..., None]
egg = egg*(1-crust)+crust*np.array([176, 128, 62.])
save('french_toast_raw.png', egg)

# Over-toasted bread top edge (B2): dark brown to charred.
char = np.array([70, 42, 22.])+25*noise((N, N), 30)[..., None]
char = np.where((noise((N, N), 16) > .45)[..., None], np.array([30, 24, 20.]), char)
save('toast_overdone.png', char)

# Smoke wisps RGBA (grey, translucent).
H, W = 1024, 512
yy, xx = np.mgrid[0:H, 0:W]
v = 1-yy/H
dens = np.zeros((H, W))
for i in range(4):
    cx = .5+.12*(i-1.5)
    axis = cx+.08*np.sin(v*7+i)*v+.04*np.sin(v*17+2*i)
    width = .05+.22*v
    dens += np.exp(-((xx/W-axis)/width)**2*2.)*(v**.6)*np.clip(1.1-v, 0, 1)
dens *= .55+.45*(noise((H, W), 60) > -.2)
alpha = np.clip(dens*.55, 0, .65)
rgba = np.dstack([np.full((H, W), 168), np.full((H, W), 168), np.full((H, W), 172), alpha*255])
save('smoke_wisps.png', rgba)
Image.fromarray((alpha/alpha.max()*255).astype(np.uint8)).save(OUT/'smoke_alpha.png')

# Gas burner flame ring RGBA (blue, emissive), seen from above under a pan.
ring = np.exp(-((r-.42)/.035)**2)
teeth = .55+.45*np.cos(np.arctan2(y-.5, x-.5)*36)**2
alpha = np.clip(ring*teeth*1.4, 0, 1)
rgba = np.dstack([40+120*alpha, 110+120*alpha, np.full((N, N), 255), alpha*255])
save('gas_flame_ring.png', rgba)

# Kitchen-scale LCD reading (initial 0 g).
img = Image.new('RGB', (512, 192), (24, 30, 28))
d = ImageDraw.Draw(img)
d.rounded_rectangle([10, 10, 502, 182], 18, fill=(164, 186, 160))
d.text((470, 96), '0 g', font=font(MONO_B, 112), fill=(20, 26, 22), anchor='rm')
d.text((26, 22), 'TARE', font=font(FONT_B, 22), fill=(40, 52, 44))
img.save(OUT/'scale_display_0g.png')

# Glass jug graduation marks RGBA.
img = Image.new('RGBA', (256, 512), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
for ml in range(50, 551, 50):
    yy0 = int(500-ml/550*470)
    long = ml % 100 == 0
    d.line([(20, yy0), (110 if long else 70, yy0)], fill=(245, 245, 245, 235), width=5 if long else 3)
    if long:
        d.text((122, yy0), f'{ml}', font=font(FONT_B, 34), fill=(245, 245, 245, 235), anchor='lm')
img.save(OUT/'jug_graduations.png')


def card_text(name, title, lines, size=(900, 600), bg=(247, 244, 236)):
    img = Image.new('RGB', size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0]-1, size[1]-1], outline=(120, 110, 96), width=6)
    d.text((40, 36), title, font=font(FONT_B, 58), fill=(40, 36, 30))
    yy0 = 130
    for ln in lines:
        d.text((44, yy0), ln, font=font(FONT, 44), fill=(52, 48, 42))
        yy0 += 64
    img.save(OUT/name)


card_text('card_A4_recipe.png', 'Recipe', ['Hot water: 300 g', 'Tolerance: +/- 10 g', 'Pour into the glass jug', 'on the scale.'])
card_text('card_C1_orders.png', 'The pass', ['One bowl on the board', 'Waiter takes it at pickup', 'Next meal: mat, then board', 'Order, times: instruction'], (900, 560))
card_text('card_C2_orders.png', 'French toast', ['4 slices, 1 flip each', 'Serve onto the plate', 'Pickups: see instruction'], (900, 420))
card_text('card_C3_seating.png', 'Next seating', ['Seat 1: next course soon', 'Seat 2: guest arriving', 'Clear, then wipe the mat'], (900, 420))

# Meal surfaces for C1 bowls.
for name, col, spots in [('meal_curry.png', (176, 104, 34), (230, 214, 170)), ('meal_rice.png', (236, 228, 206), (110, 150, 70)),
                         ('meal_soup.png', (186, 120, 70), (220, 160, 80)), ('meal_pasta.png', (222, 168, 82), (180, 50, 30))]:
    img = Image.fromarray(np.clip(np.array(col, float)+16*noise((256, 256), 24)[..., None], 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for i in range(60):
        cx, cy = rng.uniform(20, 236, 2)
        rad = rng.uniform(3, 9)
        d.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=spots)
    img.filter(ImageFilter.GaussianBlur(1.2)).save(OUT/name)

# Sauce and water stains: sauce opaque-ish, water as a faint glossy film (RGBA).
save('sauce_stain.png', np.dstack([np.full((256, 256), 120), np.full((256, 256), 46), np.full((256, 256), 18),
                                   np.full((256, 256), 255)]))
print('TEXTURES', sorted(p.name for p in OUT.iterdir()))
