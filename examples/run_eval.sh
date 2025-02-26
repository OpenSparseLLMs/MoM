MODEL_PATH=training/SlimPajama/mom-15B/checkpoint-30720

accelerate launch --multi_gpu evals/harness.py --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,lambada_standard,piqa,winogrande,wikitext \
    --output_path eval_results \
    --batch_size 32 \
    --device cuda