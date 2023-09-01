export CODE_DPATH=/home/makinot1/git/domain-invariant-generation-toy/domain_invariant_generation_toy
export DPATH=$RESULTS_DPATH/dsprites

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $DPATH \
--task train_vae

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $DPATH \
--task train_q

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $DPATH \
--task infer_z_train

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $DPATH \
--task infer_z_val

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $DPATH \
--task infer_z_test

python $CODE_DPATH/main.py \
--dataset dsprites \
--dpath $DPATH \
--task classify