# train45: five-domain fine-tuning of four RoboDojo models

The runbook is docs/five-domain-training.md. It explains every
choice against the 2026-11-11 π0.5 lab run.

| Path | Contents |
|---|---|
| `gen_instructions.py` | Per-episode prompts from the current task code (all five domains) |
| `install_artifacts.sh` | Copies `artifacts/` and role A's scripts to the paths the runs read, checking each sha256 |
| `artifacts/` | The committed inputs: prompts, norm stats of roles A, C and D; `MANIFEST.tsv` lists each sha256 and install target |
| `A_pi05/` | π0.5: config (`train_pi05_roboclock.py`), LeRobot converter, launchers, dataset check, norm-stat widening |
| `B_xvla/` | X-VLA: roboclock handler wrapper, meta builder, launcher, CPU dry test |
| `C_openwam/` | OpenWAM-α: prompt wrapper, stats merge, launcher, CPU dry test |
| `D_internw0/` | InternW0-Δ: v2.1 converter, stats, text cache and its check, launcher, eval config |

On a new node:
```bash
git pull
# fetch the 45 task dirs into $TRAIN45_DIR/data/RoboDojo (docs 3.4)
bash scripts/train45/install_artifacts.sh <A|B|C|D>
```
