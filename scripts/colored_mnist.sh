export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

export DATASET=colored_mnist
export DPATH=$RESULTS_DPATH/$DATASET

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task train_vae

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task train_classifier

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task inference \
--batch_size 2048

python $CODE_DPATH/generate_sample_prior.py \
--dpath $DPATH