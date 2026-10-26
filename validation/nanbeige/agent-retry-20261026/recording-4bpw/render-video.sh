#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# CPU rendering only. Preserve original realtime cast; GIF/MP4 accelerate playback 4x.
# agg's idle cap is 5 seconds before the 4x playback multiplier.
agg --font-family 'Noto Sans Mono' --font-size 16 --line-height 1.2 --theme asciinema --speed 4 --idle-time-limit 5 --fps-cap 15 --last-frame-duration 2 --no-loop cube-4bpw.cast cube-4bpw-4x.gif
ffmpeg -nostdin -y -threads 4 -i cube-4bpw-4x.gif -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -threads 4 -preset fast -crf 19 -pix_fmt yuv420p -movflags +faststart -an cube-4bpw-full-4x.mp4
ffmpeg -nostdin -y -ss 19.5 -i cube-4bpw-full-4x.mp4 -c:v libx264 -threads 4 -preset fast -crf 19 -pix_fmt yuv420p -movflags +faststart -an cube-4bpw-4x.mp4
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,avg_frame_rate,pix_fmt -of json cube-4bpw-4x.mp4 > video-ffprobe.json
ffmpeg -nostdin -y -ss 98.5 -i cube-4bpw-4x.mp4 -frames:v 1 video-frame.png
sha256sum cube-4bpw.cast cube-4bpw-4x.gif cube-4bpw-full-4x.mp4 cube-4bpw-4x.mp4 video-frame.png > SHA256SUMS
