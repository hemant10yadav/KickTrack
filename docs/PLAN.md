## Goal

Take a fixed-camera football video and show a marker on each player that moves with them as the video plays.

- [X] **Plan 1: Track Players in a Video**

  - [X] Add CV dependencies (Ultralytics YOLO, OpenCV) to requirements
  - [X] Install dependencies
  - [X] Sample fixed-camera football video to test with
  - [X] Script: read video frame by frame (OpenCV)
  - [X] Script: detect players in each frame (YOLO, person class)
  - [X] Script: track each player across frames with a consistent ID (ByteTrack)
  - [X] Script: draw a marker (dot/box + ID) on each tracked player
  - [X] Script: display the annotated video live as it processes (`cv2.imshow`)
  - [X] Run it end-to-end and confirm markers stay on the correct player as they move
- [X] **Plan 2: Team Classification (Jersey Colors)**

  - [X] Crop the jersey region from each tracked player's bounding box
  - [X] Extract a dominant color per player (e.g. k-means on jersey pixels)
  - [X] Cluster players into Team A / Team B / Referee based on dominant color
  - [X] Keep each track ID's team assignment stable across frames (don't flip-flop)
  - [X] Color-code markers per team (e.g. red vs blue, referee in yellow/black)
  - [X] Run end-to-end and confirm team assignment stays consistent as players move
