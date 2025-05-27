echo "PWD: $(pwd)"

cd "$(dirname "$0")/.."

echo "PWD: $(pwd)"


NUM_EPOCHS=120
DEVICE="cuda:1"
LR=1e-3
CRITERION='me'

FEATTYPE='encoder+decoder'
KERNELSIZE=1
NUM_CONVS=3
LABEL_TYPE='miou'
LOAD_SAVE=0
OUT_FIRST=32

echo "Running training (ME) $FEATTYPE $KERNELSIZE"
python ./main/train_pmoh_muds.py  --num_epochs $NUM_EPOCHS --device $DEVICE --learning_rate $LR --criterion $CRITERION --feat_type $FEATTYPE --kernel_size $KERNELSIZE --label_type $LABEL_TYPE --num_convs $NUM_CONVS --out_first $OUT_FIRST
