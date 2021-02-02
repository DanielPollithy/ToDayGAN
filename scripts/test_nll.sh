python test_NLL.py  \
    --dataroot ./datasets/robotcar  \
    --name robotcar_nll  \
    --results_dir "/net/skoll/storage/datasets/robotcar/robotcar/todaygan_new/results/nll/" \
    --n_domains 2 --phase test --which_epoch 150 --serial_test \
    --resize_or_crop none --flip_export --reconstruct --netvlad
