export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $RESULTS_DPATH/dsprites/train_vae \
--stage train_vae \
--prior_reg_mult 0.1 \
--q_mult 1

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $RESULTS_DPATH/dsprites/train_classifier \
--ckpt_fpath $RESULTS_DPATH/dsprites/train_vae/version_0/checkpoints/best.ckpt \
--stage train_classifier

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $RESULTS_DPATH/dsprites/train_q \
--ckpt_fpath $RESULTS_DPATH/dsprites/train_classifier/version_0/checkpoints/best.ckpt \
--stage train_q

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $RESULTS_DPATH/dsprites/test \
--ckpt_fpath $RESULTS_DPATH/dsprites/train_q/version_0/checkpoints/best.ckpt \
--stage test

python $CODE_DPATH/generate_from_prior.py \
--dpath $RESULTS_DPATH/dsprites/train_vae