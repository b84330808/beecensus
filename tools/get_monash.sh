#!/bin/sh
# Fetch the public Monash pollination data BeeCensus is evaluated on (CC BY 4.0, doi 10.26180/21533760)
# into $BEECENSUS_DATA (default ../data/monash). Only the needed parts of the 21 GB of archives are downloaded.
set -e
cd "$(dirname "$0")"
DATA=${BEECENSUS_DATA:-../../data/monash}
mkdir -p "$DATA/videos" "$DATA/traj"
VIDEOS=https://ndownloader.figshare.com/files/38175570   # Test_Video_Dataset.zip (10 x 10-minute videos)
YOLO=https://ndownloader.figshare.com/files/38171013     # YOLOv4_Training_and_Test_Dataset.zip (labelled frames)
TRAJ=https://ndownloader.figshare.com/files/38169609     # Insect_Trajectory_Dataset.zip (Polytrack tracks)
# zip entry offsets/sizes from: python remote_zip_ls.py $VIDEOS
get() { [ -s "$DATA/videos/$1.avi" ] || python -I robust_inflate.py $VIDEOS $2 $3 "$DATA/videos/$1.avi"; }
get T4 9135261433 940768691
get T8 2466587840 1260029314
get T7 1290855022 1175732714
get T1 5735264729 1071978957
get T6 33 1290854885
get T5 10076030228 1314031553
get T2 6807243790 1011955578   # held out: not used while tuning
get T3 7819199472 1316061857
get T9 3726617258 1022708929
get T10 4749326292 985938332
[ -d "$DATA/yolo_testing" ] || { python -I fetch_zip_prefix.py $YOLO 330000000 "$DATA/yolo_tmp" && mv "$DATA/yolo_tmp/testing" "$DATA/yolo_testing"; }
[ -d "$DATA/traj/Extracted_Tracks" ] || { curl -sL -o "$DATA/traj.zip" $TRAJ && (cd "$DATA/traj" && unzip -q ../traj.zip); }
echo done
