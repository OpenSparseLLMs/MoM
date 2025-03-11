### JRT ArXiv Table 1 ###

output_dir="run_jrt_prompt_hazy"
limit=-1


# Default and twice SWDE context length 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
  --batch-size 32 \
  -m hazyresearch/based-1b-50b \
  -m hazyresearch/mamba-1b-50b \
  -m hazyresearch/attn-1b-50bn \
  -t based_swde \
  -t based_swde_twice \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit}  \
  -p


# Default and twice FDA at context length 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
  --batch-size 32 \
  -m hazyresearch/based-1b-50b \
  -m hazyresearch/mamba-1b-50b \
  -m hazyresearch/attn-1b-50bn \
  -t based_fda \
  -t based_fda_twice \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p


# Default and twice SQUAD completion at context length 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
  --batch-size 32 \
  -m hazyresearch/based-1b-50b \
  -m hazyresearch/mamba-1b-50b \
  -m hazyresearch/attn-1b-50bn \
  -t based_squad_twice \
  -t based_squad \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
  --batch-size 32 \
  -m hazyresearch/based-1b-50b \
  -m hazyresearch/mamba-1b-50b \
  -m hazyresearch/attn-1b-50bn \
  -t based_drop_twice \
  -t based_drop \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p 


CUDA_VISIBLE_DEVICES=10,1,2,3,4,5,6,7 python launch_jrt.py \
  --batch-size 32 \
  -m hazyresearch/based-1b-50b \
  -m hazyresearch/mamba-1b-50b \
  -m hazyresearch/attn-1b-50bn \
  -t based_nq_1024_twice \
  -t based_nq_1024 \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p

 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
  --batch-size 32 \
  -m hazyresearch/based-1b-50b \
  -m hazyresearch/mamba-1b-50b \
  -m hazyresearch/attn-1b-50bn \
  -t based_triviaqa_twice \
  -t based_triviaqa \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p

