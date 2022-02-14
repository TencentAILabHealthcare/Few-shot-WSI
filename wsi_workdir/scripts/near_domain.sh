for model in fsp clp
do
    for i in {0..8}
    do
        _model=${model}_wo_${i}
        date
        python3 -u wsi_workdir/dict_construction.py \
            --model ${_model} \
            --novel_class ${i}
    done

    # RidgeClassifier and Logistic Regression
    for clf in Ridge logistic_regression
    do
        for i in {0..8}
        do
            date
            _model=${model}_wo_${i}
            for num_shot in 1 5 10
            do
                python3 -u wsi_workdir/distributed_meta_test.py \
                    --task near   \
                    --num_task 1000  \
                    --num_shots ${num_shot}   \
                    --mode linear   \
                    --model ${_model} \
                    --novel_class ${i}  \
                    --clf ${clf}

                python3 -u wsi_workdir/distributed_meta_test.py \
                    --task near   \
                    --num_task 1000  \
                    --num_shots ${num_shot}   \
                    --mode latent_aug   \
                    --model ${_model} \
                    --novel_class ${i}  \
                    --clf ${clf}
            done
        done
    done


    # nearest centroid
    for i in {0..8}
    do
        date
        _model=${model}_wo_${i}
        for num_shot in 1 5 10
        do
            python3 -u wsi_workdir/distributed_meta_test.py \
                --task near   \
                --num_task 1000  \
                --num_shots ${num_shot}   \
                --mode linear   \
                --model ${_model} \
                --novel_class ${i}  \
                --clf nearest_centroid
        done
    done
done