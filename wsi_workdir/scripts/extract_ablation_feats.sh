
# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root

model=clp_wo_78

# uncomment and modify below if you pre-train your own model.
# model_pth=work_dirs_real/wsi_selfsup/moco_v3
# _model_pth=${model_pth}/r18_bs256_ep200_wo_78/latest.pth
# python3 -u tools/extract_backbone_weights.py ${_model_pth} wsi_workdir/workdir/pretrained_weights/${model}.pth

mkdir wsi_workdir/workdir/extracted_feats/${model}

for dataset in NCT_train NCT_test
do
    date
    python3 -u wsi_workdir/extract.py \
        --pretrain wsi_workdir/workdir/pretrained_weights/${model}.pth \
        --config configs/extraction/extract_feats_${dataset}.py \
        --output wsi_workdir/workdir/extracted_feats/${model}/${dataset}.npy
done