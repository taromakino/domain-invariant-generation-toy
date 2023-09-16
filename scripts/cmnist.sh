export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

export REG_MULT=1
export DATASET=cmnist
export DPATH=$RESULTS_DPATH/$DATASET/reg_mult=$REG_MULT

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task vae \
--reg_mult $REG_MULT

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task q_z

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify_train

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify_test

python $CODE_DPATH/generate_sample_prior.py \
--dpath $DPATH

python $CODE_DPATH/generate_sample_q.py \
--dpath $DPATH