export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

export DATASET=colored_mnist
export DPATH=$RESULTS_DPATH/$DATASET

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task train_vae

python $CODE_DPATH/generate_sample_prior.py \
--dpath $DPATH

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task train_inference_encoder

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify_y_zc

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify_c_zc

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task regress_s_zc