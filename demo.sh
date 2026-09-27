#!/usr/bin/env bash
# Renders clean demo videos for the portfolio: no markers, no pass panel, no
# audio, encoded small for the web (starts playing at once), plus a poster
# image. Each clip takes as long as it runs (paced at its frame rate).
#
# The tracker runs on the whole video; only the finished demo is cut. It
# skips the first 6 s and is at most 1 minute long: a start time alone
# ("match_5 0:07") means from 0:07 for up to a minute, and an end past that
# is cut to it.
#
#   ./demo.sh match_6                 -> demos/match_6.mp4 + demos/match_6.jpg
#   ./demo.sh match_6@0:40            -> from 0:40, up to a minute
#   ./demo.sh match_6@0:40-1:10       -> only 0:40 to 1:10 of the clip
#   ./demo.sh path/to/clip.mp4        -> demos/clip.mp4
#   ./demo.sh match_4 match_6 match_7 -> one demo each, rendered one after another
#                                        (a clip that fails is reported, the rest still run)
#   ./demo.sh --publish               -> renders every clip in demos.txt and pushes them
#                                        to the gh-pages branch, served at
#                                        https://hemant10yadav.github.io/KickTrack/
#   ./demo.sh --publish --no-render   -> pushes the demos already in demos/, without
#                                        rendering them again
#
# Names are looked up in data/videos/. demos.txt lists one clip per line, with
# an optional start (or start-end) and a title for the page after "|":
#   match_6 0:40 | Real Madrid, La Liga
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
DEFAULT_START_S=6   # cut from the finished demo when no start is given
MAX_S=60            # no demo is longer than this

# "1:10" / "0:01:10" / "70" -> seconds
seconds() { awk -F: '{ s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; print s }' <<<"$1"; }

publish=false
render_clips=true
clips=()
titles=() # one per clip, "" when demos.txt gives none
if [ "${1:-}" = "--publish" ]; then
  publish=true
  shift
  if [ "${1:-}" = "--no-render" ]; then
    render_clips=false
    shift
  fi
  while IFS= read -r line; do
    title=""
    [[ $line == *"|"* ]] && title=$(sed 's/^ *//; s/ *$//; s/"/\\"/g' <<<"${line#*|}")
    read -r name range _ <<<"${line%%|*}"
    case "${name:-}" in "" | "#"*) continue ;; esac
    clips+=("$name${range:+@$range}")
    titles+=("$title")
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
  local clip=$1 range="" source name
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

  # The demo runs from the start (6 s by default) for up to MAX_S, or to
  # the given end if that comes sooner.
  local from length
  from=$(seconds "${range%%-*}")
  [ -n "$range" ] || from=$DEFAULT_START_S
  length=$MAX_S
  if [[ $range == *-* ]]; then
    length=$(awk -v f="$from" -v t="$(seconds "${range#*-}")" -v m="$MAX_S" \
      'BEGIN { l = t - f; print (l < m ? l : m) }')
  fi

  # Tracking stops where the demo ends: only the tail is dropped beforehand
  # (a stream copy, instant), so tracking still starts at the video's first
  # frame. The start is cut from the finished video.
  ffmpeg -hide_banner -loglevel error -y -i "$source" \
    -t "$(awk -v f="$from" -v l="$length" 'BEGIN { print f + l + 1 }')" \
    -c copy -an "$WORK_DIR/${name}_source.mp4"
  uv run python -m scripts.track_players "$WORK_DIR/${name}_source.mp4" \
    --output "$WORK_DIR/${name}_tracked.mp4" --hide-passes "$@"

  ffmpeg -hide_banner -loglevel error -y -ss "$from" -t "$length" \
    -i "$WORK_DIR/${name}_tracked.mp4" \
    -c:v libx264 -preset slow -crf "$DEMO_CRF" -vf "scale=$DEMO_WIDTH:-2" -pix_fmt yuv420p \
    -an -movflags +faststart "$OUT_DIR/$name.mp4"
  local middle
  middle=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT_DIR/$name.mp4" |
    awk '{print $1 / 2}')
  ffmpeg -hide_banner -loglevel error -y -ss "$middle" -i "$OUT_DIR/$name.mp4" \
    -frames:v 1 -q:v 3 "$OUT_DIR/$name.jpg"
  rm -f "$WORK_DIR/${name}_source.mp4" "$WORK_DIR/${name}_tracked.mp4"
  echo "    from ${from}s: $(du -h "$OUT_DIR/$name.mp4" | cut -f1) video, poster $OUT_DIR/$name.jpg"
}

failed=()
published=()
published_titles=()
for i in "${!clips[@]}"; do
  clip=${clips[$i]}
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
  published_titles+=("${titles[$i]:-}")
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi

if $publish; then
  # The branch holds only the site page, demos/ and .nojekyll, as one commit
  # replaced on every publish, so re-rendered videos never pile up in history.
  pages=$(mktemp -d)
  trap 'git worktree remove --force "$pages" 2>/dev/null || true' EXIT
  git worktree add --detach "$pages" >/dev/null
  git -C "$pages" checkout --orphan "$PAGES_BRANCH-publish" >/dev/null 2>&1
  git -C "$pages" rm -rfq . >/dev/null
  mkdir -p "$pages/demos"
  for name in "${published[@]}"; do
    cp "$OUT_DIR/$name.mp4" "$OUT_DIR/$name.jpg" "$pages/demos/"
  done
  # The page (site/index.html, kept on main) lists its videos from this.
  {
    printf '['
    for i in "${!published[@]}"; do
      [ "$i" -gt 0 ] && printf ','
      printf '{"name":"%s","title":"%s"}' "${published[$i]}" "${published_titles[$i]}"
    done
    printf ']\n'
  } >"$pages/demos/list.json"
  cp site/index.html "$pages/index.html"
  touch "$pages/.nojekyll"
  git -C "$pages" add -f .nojekyll index.html demos
  # --no-verify: the repo's pre-commit hooks are for code, and this branch has
  # no hook config (prek refuses to commit without one).
  git -C "$pages" commit -q --no-verify -m "demos from $(git rev-parse --short HEAD)"
  git -C "$pages" push -qf origin "HEAD:$PAGES_BRANCH"
  git -C "$pages" checkout -q --detach
  git branch -D "$PAGES_BRANCH-publish" >/dev/null
  echo "published to $PAGES_BRANCH: https://hemant10yadav.github.io/KickTrack/"
  for name in "${published[@]}"; do
    echo "  https://hemant10yadav.github.io/KickTrack/demos/$name.mp4"
  done
fi
