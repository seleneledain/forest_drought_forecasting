#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --gres=gpu:A40:1
# #SBATCH --exclusive
#SBATCH --job-name earthnet
# #SBATCH --mail-user your-mail-account@email.com
# #SBATCH --mail-type ALL
# #SBATCH -A p_da_earthnet

# $1 setting.yaml
# $2 train (test both)
# $3 code folder
# $4 data folder
# $5 test tracks list

echo $1

arr=($(cat $1 | grep "Setting:"))
setting=${arr[1]}
echo $setting


cd $3


if [ $2 == 'train' ] || [ $2 == 'both' ]; then
    if [ $setting == 'en21-std' ]; then
        echo "Making Directories"
        mkdir -p /tmp/data/train/
        echo "Copying Train"
        cp -rf $4/train/. /tmp/data/train/
    elif [ $setting == 'en21x' ]; then
        echo "Making Directories"
        mkdir -p /tmp/data/train/
        # mkdir -p /tmp/data/landcover/
        echo "Copying Train"
        cp -rf $4/train/. /tmp/data/train/
        # echo "Copying Landcover"
        # cp -rf $4/landcover/. /tmp/data/landcover/
    #elif [ $setting == "europe-veg"]; then
    #    echo "Not Implemented"
    #    exit 1
    else   
        echo "Not Implemented"
        exit 1
    fi
    
    echo "Start training"
    srun /Net/Groups/BGI/people/vbenson/miniconda3/envs/emp_a100/bin/python /Net/Groups/BGI/people/vbenson/EarthNet/earthnet-models-pytorch/scripts/train.py $1
fi

# if [ $1 == "test" || $1 == "both"]; then
#     exit 1
#     for track in $5
#     do
#         echo "Copying IID Test"
#         mkdir -p /tmp/data/${track}_test_split/ #Here copy the right track !!!!
#         cp -rf $4/${track}_test_split/. /tmp/data/${track}_test_split/
#         echo "Start testing"
#         srun python test.py $1 $track #Need getting best checkpoint!!!
#     done
# fi