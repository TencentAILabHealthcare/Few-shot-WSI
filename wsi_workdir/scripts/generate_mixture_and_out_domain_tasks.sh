# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root



# uncomment and modify below if you pre-train your own model.
# python3 -u tools/extract_backbone_weights.py \
#     work_dirs_real/classification/nct/r18_bs512_ep100_all/latest.pth \
#     wsi_workdir/workdir/pretrained_weights/fsp.pth

# python3 -u tools/extract_backbone_weights.py \
#     work_dirs_real/wsi_selfsup/moco_v3/r18_bs256_ep200_all/latest.pth \
#     wsi_workdir/workdir/pretrained_weights/clp.pth


for model in clp fsp
do
    for num_shot in 1 5 10
    do
        python3 wsi_workdir/tools/generate_task.py \
            --task mixture \
            --num_shots $num_shot \
            --num_task 1000 \
            --initialization \
            --model ${model}
    done

    for num_shot in 1 5 10 50 100
    do
        python3 wsi_workdir/tools/generate_task.py \
            --task out \
            --mode hetero \
            --num_shots ${num_shot} \
            --num_task 1000 \
            --model ${model}
    done


    for num_shot in 5 10 50 100
    do
        python3 wsi_workdir/tools/generate_task.py \
            --task out \
            --mode homo \
            --num_shots ${num_shot} \
            --num_task 1000 \
            --model ${model}
    done

done


