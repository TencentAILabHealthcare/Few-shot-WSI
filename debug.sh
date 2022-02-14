bash tools/dist_train.sh configs/wsi_selfsup/moco_v3/r18_bs256_ep200_all.py 1 --dev
for i in {0..8}
do
    bash tools/dist_train.sh configs/wsi_selfsup/moco_v3/r18_bs256_ep200_wo_{$i}.py 1 --dev
done