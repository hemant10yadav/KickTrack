#!/usr/bin/env bash
# Renders clean demo videos for the portfolio: no markers, no pass panel,
# source audio kept, encoded small for the web (starts playing at once), plus
# a poster image. Each clip takes as long as it runs (paced at its frame rate).
#
#   ./demo.sh match_6                 -> demos/match_6.mp4 + demos/match_6.jpg
#   ./demo.sh match_6@0:40-1:10       -> only 0:40 to 1:10 of the clip
#   ./demo.sh path/to/clip.mp4        -> demos/clip.mp4
#   ./demo.sh match_4 match_6 match_7 -> one demo each, rendered one after another
#                                        (a clip that fails is reported, the rest still run)
#   ./demo.sh --publish               -> renders every clip in demos.txt and pushes them
#                                        to the gh-pages branch, served at
#                                        https://hemant10yadav.github.io/KickTrack/demos/<name>.mp4
#   ./demo.sh --publish --no-render   -> pushes the demos already in demos/, without
#                                        rendering them again
#
# Names are looked up in data/videos/. demos.txt lists one clip per line, with
# an optional time range: "match_6 0:40-1:10".
#
# Any extra flags go on to scripts.track_players, for every clip:
#
#   --show                     also watch it in a live window while it renders
#                              (the window has no sound; the file does)
#   --show-markers             draw the pin + ID above each player and the rings on
#                              the ball and whoever has it
#   --analytics-dir DIR        also write stats.json, passes.json and heatmap PNGs
#                              to DIR, e.g. --analytics-dir demos/match_6_stats
#                              (with several clips, pass one clip at a time: they
#                              would all write to the same DIR)
#   --display-width PX         width the tracker renders at (default 1920)
#   --display-delay-ms MS      how far frames are held back so markers sit on the
#                              players (default 100); rarely worth changing
#   --viewer ffplay|opencv     what draws the --show window (default ffplay)
#   --model / --imgsz          a different CoreML model and the shape it was
#                              exported at (default yolo26s.mlpackage, 640,1152)
#
# The pass panel is always hidden here; for it, run scripts.track_players
# directly with --output. Video names come first, then the flags:
#   ./demo.sh match_6 --show --analytics-dir demos/match_6_stats
#   ./demo.sh --publish --show-markers
#
# Web encode: DEMO_WIDTH (default 1280) and DEMO_CRF (default 28; higher is
# smaller and blurrier) override it, e.g. DEMO_WIDTH=1920 ./demo.sh match_5
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR=demos
WORK_DIR=$OUT_DIR/.work
CLIP_LIST=demos.txt
PAGES_BRANCH=gh-pages
DEMO_WIDTH=${DEMO_WIDTH:-1280}
DEMO_CRF=${DEMO_CRF:-28}
# A trimmed clip is tracked from this many seconds before its start, then cut:
# starting cold, calibration and team colours take a while to lock on.
WARMUP_S=10

# "1:10" / "0:01:10" / "70" -> seconds
seconds() { awk -F: '{ s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; print s }' <<<"$1"; }

publish=false
render_clips=true
clips=()
if [ "${1:-}" = "--publish" ]; then
  publish=true
  shift
  if [ "${1:-}" = "--no-render" ]; then
    render_clips=false
    shift
  fi
  while read -r name range _; do
    case "$name" in "" | "#"*) continue ;; esac
    clips+=("$name${range:+@$range}")
  done <"$CLIP_LIST"
else
  while [ $# -gt 0 ] && [[ $1 != -* ]]; do
    clips+=("$1")
    shift
  done
fi
if [ ${#clips[@]} -eq 0 ]; then
  echo "usage: ./demo.sh <video name or path>[@from-to]... [track_players flags...]" >&2
  echo "       ./demo.sh --publish [track_players flags...]   (clips from $CLIP_LIST)" >&2
  exit 1
fi
mkdir -p "$WORK_DIR"

# render <name>[@from-to] [track_players flags...]
render() {
  local clip=$1 range="" source name lead=0
  shift
  if [[ $clip == *@* ]]; then
    range=${clip#*@}
    clip=${clip%@*}
  fi
  source=$clip
  [ -f "$source" ] || source="data/videos/${clip%.mp4}.mp4"
  [ -f "$source" ] || { echo "not found: $source" >&2; return 1; }
  name=$(basename "${source%.*}")
  echo "==> $source${range:+ ($range)} -> $OUT_DIR/$name.mp4"

  if [ -n "$range" ]; then
    local from to start
    from=$(seconds "${range%-*}")
    to=$(seconds "${range#*-}")
    start=$(awk -v f="$from" -v w="$WARMUP_S" 'BEGIN { print (f > w ? f - w : 0) }')
    lead=$(awk -v f="$from" -v s="$start" 'BEGIN { print f - s }')
    # Re-encoded, not stream-copied: a copy can only cut at a keyframe.
    ffmpeg -hide_banner -loglevel error -y -ss "$start" -to "$to" -i "$source" \
      -c:v h264_videotoolbox -b:v 20M -c:a aac "$WORK_DIR/$name.mp4"
    source=$WORK_DIR/$name.mp4
  fi
  uv run python -m scripts.track_players "$source" \
    --output "$WORK_DIR/${name}_tracked.mp4" --hide-passes "$@"

  ffmpeg -hide_banner -loglevel error -y -ss "$lead" -i "$WORK_DIR/${name}_tracked.mp4" \
    -c:v libx264 -preset slow -crf "$DEMO_CRF" -vf "scale=$DEMO_WIDTH:-2" -pix_fmt yuv420p \
    -c:a aac -b:a 128k -movflags +faststart "$OUT_DIR/$name.mp4"
  local middle
  middle=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT_DIR/$name.mp4" |
    awk '{print $1 / 2}')
  ffmpeg -hide_banner -loglevel error -y -ss "$middle" -i "$OUT_DIR/$name.mp4" \
    -frames:v 1 -q:v 3 "$OUT_DIR/$name.jpg"
  rm -f "$WORK_DIR/$name.mp4" "$WORK_DIR/${name}_tracked.mp4"
  echo "    $(du -h "$OUT_DIR/$name.mp4" | cut -f1) video, poster $OUT_DIR/$name.jpg"
}

failed=()
published=()
for clip in "${clips[@]}"; do
  name=${clip%@*}
  name=$(basename "${name%.*}")
  if $render_clips; then
    render "$clip" "$@" || { failed+=("$clip"); continue; }
  elif [ ! -f "$OUT_DIR/$name.mp4" ] || [ ! -f "$OUT_DIR/$name.jpg" ]; then
    echo "not rendered yet: $OUT_DIR/$name.mp4 (run without --no-render)" >&2
    failed+=("$clip")
    continue
  fi
  published+=("$name")
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi

if $publish; then
  # The branch holds only demos/ and .nojekyll, as one commit replaced on every
  # publish, so re-rendered videos never pile up in git history.
  pages=$(mktemp -d)
  trap 'git worktree remove --force "$pages" 2>/dev/null || true' EXIT
  git worktree add --detach "$pages" >/dev/null
  git -C "$pages" checkout --orphan "$PAGES_BRANCH-publish" >/dev/null 2>&1
  git -C "$pages" rm -rfq . >/dev/null
  mkdir -p "$pages/demos"
  for name in "${published[@]}"; do
    cp "$OUT_DIR/$name.mp4" "$OUT_DIR/$name.jpg" "$pages/demos/"
  done
  touch "$pages/.nojekyll"
  git -C "$pages" add -f .nojekyll demos
  # --no-verify: the repo's pre-commit hooks are for code, and this branch has
  # no hook config (prek refuses to commit without one).
  git -C "$pages" commit -q --no-verify -m "demos from $(git rev-parse --short HEAD)"
  git -C "$pages" push -qf origin "HEAD:$PAGES_BRANCH"
  git -C "$pages" checkout -q --detach
  git branch -D "$PAGES_BRANCH-publish" >/dev/null
  echo "published to $PAGES_BRANCH:"
  for name in "${published[@]}"; do
    echo "  https://hemant10yadav.github.io/KickTrack/demos/$name.mp4"
  done
fi
