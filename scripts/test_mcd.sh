python test_MC_Dropout.py  \
    --dataroot ./datasets/robotcar  \
    --name robotcar_dropout  \
    --results_dir /net/skoll/storage/datasets/robotcar/robotcar/todaygan_new/results/mcd/samples/30 \
    --n_domains 2 --phase test --which_epoch 150 --serial_test --use_dropout \
    --resize_or_crop none --monte_carlo_samples 30  --flip_export
