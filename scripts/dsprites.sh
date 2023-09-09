export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

export DATASET=dsprites
export DPATH=$RESULTS_DPATH/$DATASET

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task train_vae \
--z_norm_mult 0.001

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task train_zc_sampler

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task inference

python $CODE_DPATH/generate_sample_prior.py \
--dpath $DPATH