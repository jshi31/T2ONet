BASEDIR=$(dirname "$0")
cd $BASEDIR
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='.' python demo/seq2seqL1.py --trial 1 --img "$1" --request "`echo ${@:3}`" --multi_img "$2"
echo "finished!"
