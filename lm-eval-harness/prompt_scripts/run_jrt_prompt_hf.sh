### JRT ArXiv Table 1 ###

output_dir="run_jrt_prompt_hf"
limit=-1

# Default and twice SWDE context length 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_hf.py \
  --batch-size 32 \
  -m "fla-hub/gla-1.3B-100B" \
  -m "fla-hub/gla-2.7B-100B" \
  -m "state-spaces/mamba-130m" \
  -m "state-spaces/mamba-370m" \
  -m "state-spaces/mamba-1.4b" \
  -m "state-spaces/mamba-2.8b" \
  -m "state-spaces/mamba2-130m" \
  -m "state-spaces/mamba2-370m" \
  -m "state-spaces/mamba2-1.3b" \
  -m "state-spaces/mamba2-2.7b" \
  -t based_swde \
  -t based_swde_twice \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p


# Default and twice FDA at context length 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_hf.py \
  --batch-size 32 \
  -m "fla-hub/gla-1.3B-100B" \
  -m "fla-hub/gla-2.7B-100B" \
  -m "state-spaces/mamba-130m" \
  -m "state-spaces/mamba-370m" \
  -m "state-spaces/mamba-1.4b" \
  -m "state-spaces/mamba-2.8b" \
  -m "state-spaces/mamba2-130m" \
  -m "state-spaces/mamba2-370m" \
  -m "state-spaces/mamba2-1.3b" \
  -m "state-spaces/mamba2-2.7b" \
  -t based_fda \
  -t based_fda_twice \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p


# Default and twice SQUAD completion at context length 1000
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 python launch_hf.py \
  --batch-size 32 \
  -m "fla-hub/gla-1.3B-100B" \
  -m "fla-hub/gla-2.7B-100B" \
  -m "state-spaces/mamba-130m" \
  -m "state-spaces/mamba-370m" \
  -m "state-spaces/mamba-1.4b" \
  -m "state-spaces/mamba-2.8b" \
  -m "state-spaces/mamba2-130m" \
  -m "state-spaces/mamba2-370m" \
  -m "state-spaces/mamba2-1.3b" \
  -m "state-spaces/mamba2-2.7b" \
  -t based_squad_twice \
  -t based_squad \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p


CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 python launch_hf.py \
  --batch-size 32 \
  -m "fla-hub/gla-1.3B-100B" \
  -m "fla-hub/gla-2.7B-100B" \
  -m "state-spaces/mamba-130m" \
  -m "state-spaces/mamba-370m" \
  -m "state-spaces/mamba-1.4b" \
  -m "state-spaces/mamba-2.8b" \
  -m "state-spaces/mamba2-130m" \
  -m "state-spaces/mamba2-370m" \
  -m "state-spaces/mamba2-1.3b" \
  -m "state-spaces/mamba2-2.7b" \
  -t based_drop_twice \
  -t based_drop \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p 


CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 python launch_hf.py \
  --batch-size 32 \
  -m "fla-hub/gla-1.3B-100B" \
  -m "fla-hub/gla-2.7B-100B" \
  -m "state-spaces/mamba-130m" \
  -m "state-spaces/mamba-370m" \
  -m "state-spaces/mamba-1.4b" \
  -m "state-spaces/mamba-2.8b" \
  -m "state-spaces/mamba2-130m" \
  -m "state-spaces/mamba2-370m" \
  -m "state-spaces/mamba2-1.3b" \
  -m "state-spaces/mamba2-2.7b" \
  -t based_nq_1024_twice \
  -t based_nq_1024 \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p


CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 python launch_hf.py \
  --batch-size 32 \
  -m "fla-hub/gla-1.3B-100B" \
  -m "fla-hub/gla-2.7B-100B" \
  -m "state-spaces/mamba-130m" \
  -m "state-spaces/mamba-370m" \
  -m "state-spaces/mamba-1.4b" \
  -m "state-spaces/mamba-2.8b" \
  -m "state-spaces/mamba2-130m" \
  -m "state-spaces/mamba2-370m" \
  -m "state-spaces/mamba2-1.3b" \
  -m "state-spaces/mamba2-2.7b" \
  -t based_triviaqa_twice \
  -t based_triviaqa \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p

