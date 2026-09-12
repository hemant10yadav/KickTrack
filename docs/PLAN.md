## Goal
Take a fixed-camera football video and show a marker on each player that moves with them as the video plays.

- [x] **Plan 1: Track Players in a Video**
  - [x] Add CV dependencies (Ultralytics YOLO, OpenCV) to requirements
  - [x] Install dependencies
  - [x] Sample fixed-camera football video to test with
  - [x] Script: read video frame by frame (OpenCV)
  - [x] Script: detect players in each frame (YOLO, person class)
  - [x] Script: track each player across frames with a consistent ID (ByteTrack)
  - [x] Script: draw a marker (dot/box + ID) on each tracked player
  - [x] Script: display the annotated video live as it processes (`cv2.imshow`)
  - [x] Run it end-to-end and confirm markers stay on the correct player as they move
