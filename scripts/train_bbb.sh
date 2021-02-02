python train_BBB.py --dataroot ./datasets/robotcar \
  --name robotcar_BBB_kl_0_001 \
  --n_domains 2 \
  --niter 75 \
  --niter_decay 75 \
  --loadSize 512 \
  --fineSize 384 \
  --checkpoints_dir "/net/skoll/storage/datasets/robotcar/robotcar/todaygan_new/bbb_150/kl/0.001" \
  --kl_beta 0.001
