datename=$(date +%Y%m%d-%H%M%S)
name=satcle
logging_dir=outputs/$name/$datename
mkdir -p $logging_dir
cp -r satcle $logging_dir/satcle
CUDA_VISIBLE_DEVICES=0 python satcle/main_satcle.py \
        --time $datename \
        |& tee $logging_dir/logs_$name\_$datename.txt 2>&1
