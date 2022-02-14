# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root


#### FSP
for i in {0..8}
do
    _model=fsp_wo_${i}
    for num_shot in 1 5 10
    do
        python3 -u wsi_workdir/tools/generate_task.py \
            --task near \
            --initialization \
            --num_shots ${num_shot} \
            --num_task 1000 \
            --model ${_model} \
            --novel_class ${i}
    done
        
done




#### CLP
for i in {0..8}
do
    _model=clp_wo_${i}
    for num_shot in 1 5 10
    do
        python3 -u wsi_workdir/tools/generate_task.py \
            --task near \
            --num_shots ${num_shot} \
            --num_task 1000 \
            --model ${_model} \
            --novel_class ${i}
    done
done