# Script/command for turning image sequence of recorded path tracings
# into an mp4 video.
# Copyright (C) Bill Varcho

ffmpeg -r 24 -i ../record/img_%05d.png -vb 20M -vcodec mpeg4 out.mp4 && open out.mp4
