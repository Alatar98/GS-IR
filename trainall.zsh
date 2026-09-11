#!/usr/bin/env zsh

#scenes=(bonsai counter flowers garden kitchen room stump treehill)
scenes=(bonsai counter flowers garden room treehill)



for scene in $scenes; do
  echo "========= Running $scene ========="

  python train.py -m outputs/$scene/ -s ../Datasets/nerf/$scene/ \
    --iterations 30000 -i images_4 -r 1 --eval

  python baking.py -m outputs/$scene/ \
    --checkpoint outputs/$scene/chkpnt30000.pth \
    --bound 16.0 --occlu_res 256 --occlusion 0.4

  python train.py -m outputs/$scene/ -s ../Datasets/nerf/$scene/ \
    --start_checkpoint outputs/$scene/chkpnt30000.pth \
    --iterations 40000 -i images_4 -r 1 --eval --metallic --indirect

  python render_albedo.py -m outputs/$scene/ \
    --checkpoint outputs/$scene/chkpnt40000.pth

  python train.py -m outputs/$scene/ -s ../Datasets/nerf/$scene/ \
    --start_checkpoint outputs/$scene/chkpnt40000.pth \
    --iterations 60000 -i images_4 -r 1 --eval --metallic --indirect

  python render.py -m outputs/$scene/ \
    --checkpoint outputs/$scene/chkpnt60000.pth

  python train.py -m outputs/$scene/ -s ../Datasets/nerf/$scene/ \
    --start_checkpoint outputs/$scene/chkpnt60000.pth \
    --iterations 80000 -i images_4 -r 1 --eval --metallic --indirect
  
  python baking.py -m outputs/$scene/ \
    --checkpoint outputs/$scene/chkpnt80000.pth \
    --bound 16.0 --occlu_res 256 --occlusion 0.4
  
  python tain.py -m outputs/$scene/ -s ../Datasets/nerf/$scene/ \
    --start_checkpoint outputs/$scene/chkpnt80000.pth \
    --iterations 100000 -i images_4 -r 1 --eval --metallic --indirect

  python render.py -m outputs/$scene/ \
    --checkpoint outputs/$scene/chkpnt100000.pth

  python train.py -m outputs/$scene/ -s ../Datasets/nerf/$scene/ \
    --start_checkpoint outputs/$scene/chkpnt100000.pth \
    --iterations 150000 -i images_4 -r 1 --eval --metallic --indirect

  python render.py -m outputs/$scene/ \
    --checkpoint outputs/$scene/chkpnt150000.pth
done
