# uncomment and modify below if you pre-train your own model.
# python3 -u tools/extract_backbone_weights.py \
#     work_dirs_real/classification/nct/r18_bs512_ep100_all/latest.pth \
#     wsi_workdir/workdir/pretrained_weights/fsp.pth

# python3 -u tools/extract_backbone_weights.py \
#     work_dirs_real/wsi_selfsup/moco_v3/r18_bs256_ep200_all/latest.pth \
#     wsi_workdir/workdir/pretrained_weights/clp.pth


for model in clp fsp
do
    mkdir wsi_workdir/workdir/extracted_feats/${model}
    for ds in NCT_train LC PAIP_train PAIP_test 
    do
        python3 -u wsi_workdir/extract.py \
            --pretrain wsi_workdir/workdir/pretrained_weights/${model}.pth \
            --config configs/extraction/extract_feats_${ds}.py \
            --output wsi_workdir/workdir/extracted_feats/${model}/${ds}.npy
    done
done
