#!/usr/bin/env zsh

#scenes=(bonsai counter flowers garden kitchen room stump treehill)

NAME="lego_rig"

echo "========= Running $NAME GSIR ========="



python train.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --iterations 30000 -r 1 --eval

python render_albedo.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt30000.pth

python baking.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt30000.pth \
    --bound 16.0 --occlu_res 256 --occlusion 0.4

python train.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}/chkpnt30000.pth \
    --iterations 40000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt40000.pth

python train.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}/chkpnt40000.pth \
    --iterations 60000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt60000.pth

python train.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}/chkpnt60000.pth \
    --iterations 80000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt80000.pth

python train.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}/chkpnt80000.pth \
    --iterations 100000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt100000.pth

python train.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}/chkpnt100000.pth \
    --iterations 150000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}/chkpnt150000.pth


echo "========= Running ${NAME} latebake GSIR ========="

python train.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --iterations 80000  -r 1 --eval --pbr_iteration 80000

python render_albedo.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}_late_bake/chkpnt80000.pth

python baking.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}_late_bake/chkpnt80000.pth \
    --bound 16.0 --occlu_res 256 --occlusion 0.4

python train.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}_late_bake/chkpnt80000.pth \
    --iterations 100000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}_late_bake/chkpnt100000.pth

python train.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --start_checkpoint outputs/${NAME}_late_bake/chkpnt100000.pth \
    --iterations 150000  -r 1 --eval --metallic --indirect

python render_albedo.py -m outputs/${NAME}_late_bake/ -s ../Datasets/tobi/${NAME}/ \
    --checkpoint outputs/${NAME}_late_bake/chkpnt150000.pth