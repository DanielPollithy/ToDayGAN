python train_NLL.py  \
    --dataroot ./datasets/robotcar  \
    --name robotcar_nll  \
    --n_domains 2  \
    --niter 75  --niter_decay 75  \
    --loadSize 512  --fineSize 384 \
    --lambda_cycle 0.000001 \
    --checkpoints_dir "/net/skoll/storage/datasets/robotcar/robotcar/todaygan_new/nll/"
