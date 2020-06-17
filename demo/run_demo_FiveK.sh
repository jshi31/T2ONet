BASEDIR=$(dirname "$0")
cd $BASEDIR
CUDA_VISIBLE_DEVICES=1 PYTHONPATH='.' /u/jshi31/anaconda3/bin/python core/demo_seq2seqL1.py --trial 1 --img "$1" --request "`echo ${@:3}`" --multi_img "$2"
echo "finished!"
