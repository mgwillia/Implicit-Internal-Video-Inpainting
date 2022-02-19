#!/bin/bash

#SBATCH --job-name=fdvd_test
#SBATCH --output=outfiles/fdvd_test.out.%j
#SBATCH --error=outfiles/fdvd_test.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

SIGMAS=("10" "20" "30" "40" "50")
DAVIS_FOLDERS=("aerobatics" "car-race" "carousel" "cats-car" "chamaleon" "deer" "giant-slalom" "girl-dog" "golf" "guitar-violin" "gym" "helicopter" "horsejump-stick" "hoverboard" "lock" "man-bike" "monkeys-trees" "mtb-race" "orchid" "people-sunset" "planes-crossing" "rollercoaster" "salsa" "seasnake" "skate-jump" "slackline" "subway" "tandem" "tennis-vest" "tractor")

srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS;"
srun bash -c "./msrsync -p 4 /fs/vulcan-projects/action_augment_hao/gnerv/data/DAVIS/JPEGImages/480p/ /scratch0/mgwillia/DAVIS/;"
srun bash -c "ls /scratch0/mgwillia;"
srun bash -c "ls /scratch0/mgwillia/DAVIS;"

srun bash -c "hostname;"
for SIGMA in ${SIGMAS[@]}; do
    for FOLDER in ${DAVIS_FOLDERS[@]}; do
        srun bash -c "echo $lr;"
        srun bash -c "mkdir -p /vulcanscratch/mgwillia/fastdvdnet_results/$SIGMA/$FOLDER;"
        srun bash -c "python test_fastdvdnet.py --test_path /scratch0/mgwillia/DAVIS/$FOLDER --noise_sigma $SIGMA --save_path /vulcanscratch/mgwillia/fastdvdnet_results/$SIGMA/$FOLDER/;"
    done
done
