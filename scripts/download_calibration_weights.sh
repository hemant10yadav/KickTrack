#!/usr/bin/env bash
# Downloads PnLCalib's pretrained single-view keypoint/line detection
# weights (SoccerNet-pretrained). Gitignored -- see CLAUDE.md known
# gotchas: GitHub release downloads can be slow/unstable on this network.
set -euo pipefail
mkdir -p weights/pnlcalib
curl -L -o weights/pnlcalib/SV_kp \
  https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o weights/pnlcalib/SV_lines \
  https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
echo "Downloaded weights/pnlcalib/{SV_kp,SV_lines}"
