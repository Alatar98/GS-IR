#!/usr/bin/env zsh

datasets=(nerf lumigauss_clean)
scenes_nerf=(bonsai counter flowers garden room treehill)
scenes_lumi=(st lwp lk2 trevi)

checkpoints1=(30000 40000 60000 80000 100000 150000)
checkpoints2=(80000 100000 150000)
last_checkpoint=0


for dataset in $datasets; do
    if [[ $dataset == "nerf" ]]; then
        continue
        scenes=($scenes_nerf)
        train_options="-i=images_4"
    elif [[ $dataset == "lumigauss_clean" ]]; then
        scenes=($scenes_lumi)
        train_options=""
    fi

    for scene in $scenes; do
        if [[ $scene == "trevi" ]]; then
            resolution="2"
        else
            resolution="1"
        fi
        echo "========= Running $scene ========="
        last_checkpoint=0
        for checkpoint in $checkpoints1; do
            if [[ $last_checkpoint == 0 ]]; then
                python train.py -m outputs/$scene/ -s ../Datasets/$dataset/$scene/ \
                --iterations $checkpoint $train_options -r $resolution --eval

                python baking.py -m outputs/$scene/ \
                --checkpoint outputs/$scene/chkpnt${checkpoint}.pth \
                --bound 16.0 --occlu_res 256 --occlusion 0.4
            else
                python train.py -m outputs/$scene/ -s ../Datasets/$dataset/$scene/ \
                --start_checkpoint outputs/$scene/chkpnt${last_checkpoint}.pth \
                --iterations $checkpoint $train_options -r $resolution --eval --metallic --indirect
            fi
            last_checkpoint=$checkpoint

            python render_albedo.py -m outputs/$scene/ \
            --checkpoint outputs/$scene/chkpnt$((checkpoint)).pth
        done

        last_checkpoint=0
        outputs_name="_late_bake"
        for checkpoint in $checkpoints2; do
            if [[ $last_checkpoint == 0 ]]; then
                python train.py -m outputs/$scene$outputs_name/ -s ../Datasets/$dataset/$scene/ \
                --iterations $checkpoint $train_options -r $resolution --eval --pbr_iteration 80000
                
                python baking.py -m outputs/$scene$outputs_name/ \
                --checkpoint outputs/$scene$outputs_name/chkpnt${checkpoint}.pth \
                --bound 16.0 --occlu_res 256 --occlusion 0.4
            else
                python train.py -m outputs/$scene$outputs_name/ -s ../Datasets/$dataset/$scene/ \
                --start_checkpoint outputs/$scene$outputs_name/chkpnt${last_checkpoint}.pth \
                --iterations $checkpoint $train_options -r $resolution --eval --metallic --indirect --pbr_iteration 80000
            fi
            last_checkpoint=$checkpoint

            python render_albedo.py -m outputs/$scene$outputs_name/ \
            --checkpoint outputs/$scene$outputs_name/chkpnt$((checkpoint)).pth
        done
    done
done