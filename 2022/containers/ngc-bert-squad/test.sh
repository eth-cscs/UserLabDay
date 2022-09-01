module load sarus
sarus run --tty --mount=type=bind,source=$SCRATCH,destination=$SCRATCH \
          nvcr.io/nvidia/pytorch:22.08-py3 \
          bash -c '
          cd $SCRATCH/UserLabDay/2022/containers/ngc-bert-squad
          . deepspeed-env/bin/activate;
          python bert_squad_deepspeed_train_test.py --model-file model_finetuned_deepspeed \
                                                    --start-sample 24100 \
                                                    --num-test-samples 10'
