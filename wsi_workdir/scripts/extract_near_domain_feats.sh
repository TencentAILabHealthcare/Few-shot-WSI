
#### FSP
model_pth=work_dirs_real/classification/nct
for i in {0..8}
do
    _model=fsp_wo_${i}

    # uncomment and modify below if you pre-train your own models.
    # _model_pth=${model_pth}/r18_bs512_ep100_wo_${i}/latest.pth
    # python3 -u tools/extract_backbone_weights.py ${_model_pth} wsi_workdir/workdir/pretrained_weights/${_model}.pth
    
    mkdir wsi_workdir/workdir/extracted_feats/${_model}
    for dataset in NCT_train NCT_test
    do
        python3 -u wsi_workdir/extract.py \
            --pretrain wsi_workdir/workdir/pretrained_weights/${_model}.pth \
            --config configs/extraction/extract_feats_${dataset}.py \
            --output wsi_workdir/workdir/extracted_feats/${_model}/${dataset}.npy
    done
done




#### CLP
model_pth=work_dirs_real/wsi_selfsup/moco_v3
for i in {0..8}
do
    _model=clp_wo_${i}

    # uncomment and modify below if you pre-train your own model.
    # _model_pth=${model_pth}/r18_bs256_ep200_wo_${i}/latest.pth
    # python3 -u tools/extract_backbone_weights.py ${_model_pth} wsi_workdir/workdir/pretrained_weights/${_model}.pth

    mkdir wsi_workdir/workdir/extracted_feats/${_model}
    for dataset in NCT_train NCT_test
    do
        python3 -u wsi_workdir/extract.py \
            --pretrain wsi_workdir/workdir/pretrained_weights/${_model}.pth \
            --config configs/extraction/extract_feats_${dataset}.py \
            --output wsi_workdir/workdir/extracted_feats/${_model}/${dataset}.npy
    done
done