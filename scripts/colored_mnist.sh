export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

export DATASET=colored_mnist
export DPATH=$RESULTS_DPATH/$DATASET

export Z_NORM_MULT=0.001

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task vae \
--z_norm_mult $Z_NORM_MULT

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task q_z

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task infer_z \
--batch_size 2048

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify

python $CODE_DPATH/generate_sample_prior.py \
--dpath $DPATH

python $CODE_DPATH/generate_sample_q.py \
--dpath $DPATH

python $CODE_DPATH/generate_e_invariant.py \
--dpath $DPATH