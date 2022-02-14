# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root

mkdir wsi_workdir/workdir/tasks/NCT_78_aug
for num_shot in 1 5 10
do
    python3 -u wsi_workdir/tools/generate_aug_NCT78_task.py \
        --task NCT78 \
        --num_shots ${num_shot} \
        --num_task 300 
done
        