bash train.sh \
  nodes=4 \
  ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
  gpus=8 \
  type=mom \
  lr=3e-4 \
  steps=30720 \
  batch=8 \
  update=1 \
  warmup=1024 \
  context=2048 \
  path=SlimPajama/mom-15B \
  project=SlimPajama \
  model=configs/mom.json \
  tokenizer=fla-hub/gla-1.3B-100B \
  data=SlimPajama-627B \
  cache=data/chunk1/train