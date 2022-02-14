
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


