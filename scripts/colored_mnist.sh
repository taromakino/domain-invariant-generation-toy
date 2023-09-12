export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy

export DATASET=colored_mnist
export DPATH=$RESULTS_DPATH/$DATASET

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task vae

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task agg_posterior

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task infer_z \
--inference_stage train \
--batch_size 2048

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task infer_z \
--inference_stage val \
--batch_size 2048

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task infer_z \
--inference_stage test \
--batch_size 2048

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify

python $CODE_DPATH/main.py \
--dataset $DATASET \
--dpath $DPATH \
--task classify \
--is_spurious

python $CODE_DPATH/generate_sample_prior.py \
--dpath $DPATH