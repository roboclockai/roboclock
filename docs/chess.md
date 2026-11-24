# Chess move and clock benchmark

> **Removed 2026-11-21.** The standalone single-Franka `chess` environment (`--env chess`) was
> removed so every domain uses only the Dual ARX X5. Its rules, task model, config and scene
> (`src/roboclock/envs/chess/`) remain as the library behind sports A1 (`--env sports`); this page is
> kept for the task design and rule history. Git history has the old env (last at `20633da`).

The `chess` domain implements the supplied task: **"Make a chess move, then
press the clock. Place the piece, then press the clock within D s of the
opponent's press. Chessboard + cylinder pieces + clock buttons."**

The user confirmed that D starts at the opponent's press, varies with difficulty,
and the environment supplies a legal move. The implementation uses a complete
standard chess position and rules engine, with one robot turn per episode.
It evaluates manipulation and timing; it does not evaluate move selection.

## Run

```bash
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
roboclock tasks --env chess
roboclock plan --env chess --difficulty all --seeds 10
roboclock smoke --env chess --steps 120
roboclock render --env chess --output results/chess/board.png --usd
roboclock eval --env chess --policy oracle --oracle --difficulty easy --seeds 1
roboclock validate --env chess --difficulty all --seeds 3 --seconds 60 --images
python -m roboclock.envs.chess.videos --output results/chess/demo.mp4 --seed 0
```

Isaac Sim requires accepting NVIDIA's license on first use. The Franka USD is
loaded from NVIDIA's asset service and may require network access. The diagnostic
oracle uses ground-truth poses and contacts; ordinary policies see RGB, optional
depth, robot proprioception, instruction, and simulation time. They receive no
future opponent timestamp, contacts, or privileged scoring state.

## Rules and timing

- `chess==1.11.2` validates FEN, move history, side to move, king safety, captures,
  castling rights, en passant, all four promotions, and terminal game outcomes.
  History is replayed rather than discarded, preserving repetition claims.
- The default position is the standard starting position after `e4 e5`, and the
  assigned robot move is `Nf3` (`g1f3`). Edit `episode.initial_fen`, `history`, and
  `move` in a copy of the configuration and pass it with `--config`.
  `move: null` samples reproducibly from legal moves. An invalid or already
  terminal starting position is rejected. Chess960 is outside this domain.
- Captures require placing the captured cylinder on the capture tray before
  moving the attacking piece. En passant removes the pawn from its actual square.
  Castling requires relocating both king and rook. All other pieces must still
  occupy their assigned squares at completion.
- Cylinders carry P/N/B/R/Q/K labels and light/dark body colors. Promotion uses
  an electronic identity label on the same physical token, updated after the
  promoted pawn is placed. This is an explicit token abstraction: no replacement
  body is teleported onto the board. Tournament touch-move adjudication and
  physical spare-piece exchange are not modeled.
- The opponent move is part of the initial recorded legal history. Its clock
  button is depressed by an environment-controlled prismatic drive after a
  seeded delay; no second robot or human hand is simulated. The measured button
  depression triggers the visible turn indicator and starts D.
- A required piece must have a measured bilateral grasp, be fully inside its
  target square, upright, supported by the board/tray, released from robot
  contact, and stable for 0.2 s. A legal placement must remain valid at clock press.
- The robot clock uses a spring-return prismatic button with 6 mm travel. A press
  needs at least 3 mm measured travel plus robot contact for 25 ms. Timing uses
  the first qualifying tick, with inclusive `press <= opponent_press + D`.
  Debouncing does not move the timing timestamp later.
- Moving before the opponent's press or pressing before valid placement latches
  a violation. Missing D also latches a violation; a late physical completion can
  still be reported separately. Wrong placement never commits the logical move.
- Easy/medium/hard use provisional D values of 30/18/10 s. Geometry, moves and
  the opponent delay are identical at a given seed across difficulties. These
  values are implementation assumptions, configurable independently of physics.
  Episode budget is 60 s; physics is 120 Hz and control is 30 Hz.
- Checkmate, stalemate, insufficient material, automatic 75-move/fivefold rules,
  and optional claimed 50-move/threefold draws are reported by the rules engine.
  FEN alone cannot recover earlier repetition history; supply `history` for that.
  This is a single-turn temporal benchmark, not a full two-player game UI or a
  tournament total-time/increment clock.

## Assets and physical representation

The bundled board and clock reuse the **Chess Set and Board** CC0 pack:
<https://3dassets.dev/packs/classic-chess-essentials>. Original URLs, license,
AI-generated source designation, bounding boxes and SHA-256 values are recorded
in `src/roboclock/envs/chess/assets/sources.json`. The files are existing external
models, not newly generated assets. Their quantized GLB coordinates are decoded
with normalized accessors, transformed from Y-up to Z-up, and checked in meters.

The walnut board is 460 mm wide with 50 mm squares and a 19 mm playing surface.
The dual analog clock shell is 160 mm wide. Clock hands in the source mesh are
decorative and static; the colored indicator conveys active turn. The source
button meshes are replaced by visible dynamic buttons with actual travel.

Franka and the Poly Haven studio HDR reuse the project's existing asset paths.
The cylinder pieces are authored task geometry: 24 mm diameter, 45 mm tall,
35 g each. The board and clock shell use conservative fixed cuboid colliders;
loose cylinders use rigid dynamics, gravity, contact friction and joint-controlled
robot manipulation. No object attachment or pose edits fabricate grasp success.

Materials preserve the pack's walnut/ivory/metal base colors, metallic and
roughness values using UsdPreviewSurface. The asset pack contains no texture
maps. Generated identity/coordinate labels are sRGB text textures, with explicit
UVs and no hidden timing information. The shared HDR plus area lights illuminate
the board. Table and capture support are authored fixtures. No laboratory clutter
is added to the chessboard.

The Notion reference supplied with the request could not be read: the connected
Notion tool returned `object_not_found`/404. Its asset recommendations have not
been inferred. The public assets above are used pending accessible source links.

## Validation and artifacts

`tests/envs/chess/test_chess.py` covers legal and illegal moves, special moves,
history/draw handling, order violations, support/release/settling, timing boundaries,
missing pieces, reproducibility, asset scale and CLI planning. Unit tests verify
logic and do not establish that a robot physically succeeds.

`validate` writes a manifest (configuration, source hashes, asset hashes and plan),
per-control-step traces, events, physical/temporal metrics, and optional actual
camera frames. Image export checks that capture does not advance physics time.
`eval` records full episode results through the shared evaluator. Diagnostic
controller failures are retained.

The rules dependency is python-chess, distributed under GPL-3.0-or-later; retain
its license obligations when redistributing a combined application. Asset CC0
licensing is separate from NVIDIA simulator/robot asset terms.

An actual successful default-move recording is available at
`results/chess/chess_move_clock_labeled.mp4`; its JSON sidecar includes event
times, the result, source/asset hashes and zero-delta capture checks. See the
validation report for the exact tested scope.
