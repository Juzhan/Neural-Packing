python main.py \
--box-num 20 --box-range 10 80 --container-size 100 100 100 \
--train 0 \
--test-num 100 \
--model 'tnpp' \
--prec-type 'attn' \
--fact-type 'tap_fake' \
--data-type 'rand' \
--ems-type 'ems-id' \
--stable-rule 'hard_after_pack' \
--rotate-axes 'x' 'y' 'z' \
--hidden-dim 128 \
--world-type 'real' \
--container-type 'single' \
--pack-type 'last' \
--stable-predict 1 \
--reward-type 'C' \
--note 'for_test' \
--resume-path "checkpoints/xyz_100_100_10-80_20_tap_fake_rand/ppo_tnpp_attn_ems-id-stair_hard_after_pack_train_comp_a2c_pred/policy.pth" \

