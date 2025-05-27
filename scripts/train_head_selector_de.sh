echo "#### STARTING DATA PREPARATION ####"

dataset='dynamicearthnet'
experiment='multihead_multiutae_learned'
return_all_heads=1

python hs_data_prep.py dataset=$dataset experiment=$experiment return_all_heads=$return_all_heads

echo "#### STARTING HEAD SELECTOR TRAINING ####"

sh headselector/scripts/headwise_pmoh_de.sh

