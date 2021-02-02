python test_BBB.py  \
    --dataroot ./datasets/robotcar  \
    --name robotcar_BBB_kl_0_001  \
    --results_dir /net/skoll/storage/datasets/robotcar/robotcar/todaygan_new/results/bbb_150/kl/0.001/samples/30/ \
    --n_domains 2 --phase test --which_epoch 150 --serial_test \
    --resize_or_crop none --monte_carlo_samples 30  --flip_export \
    --netvlad