python train.py  \
    --dataroot ./datasets/robotcar  \
    --name robotcar_dropout  \
    --n_domains 2  \
    --niter 75  --niter_decay 75  \
    --loadSize 512  --fineSize 384 \
    --use_dropout