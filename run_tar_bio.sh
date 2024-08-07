accelerate launch --config_file $ExACCEL_CONFIG tar.py \
--trainer_type tar_trainer \
--max_steps 750 \
--tar_num_tasks_sampled 1 \
--tar_tamper_resistance_loss_type max_entropy \
--tar_inner_loop_steps 64 \
--retain_representations \
--unbounded \
--use_weighting_schedule \
--tar_tamper_resistance_grad_scale 4.0 \
--tar_retain_scale 1.0 \
--schedule_lambda 0.0625 \
--warmup_steps 32 \
--lr 2e-05 \
--adversary_lr_samples 2e-6,2e-5,4e-5 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--adversary_dist_types pile-bio:0.33,camel-bio:0.33,retain_forget_switch:0.33 \
--switching_point_coeffs alpha:6.0,beta:3.0 \
--adversary_lr_schedulers constant:1.0 \
--inner_optimizer_warmup_steps 20 \
--tar_inner_loop_subsample 4 \
--tar_adversary_batch_size 4 \
--base_model_name lapisrocks/Llama-3-8B-Instruct-Random-Mapped-Bio \
--subject bio \
--base llama3 \
--new_model_name Llama-3-8B-Instruct-TAR-Bio