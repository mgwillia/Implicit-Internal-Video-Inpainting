#!/bin/bash

#SBATCH --job-name=blackswan
#SBATCH --output=outfiles/blackswan.out.%j
#SBATCH --error=outfiles/blackswan.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

DAVIS_FOLDERS=("blackswan")

srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS;"
srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS_ANNOS;"
srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS_MASKS;"
srun bash -c "./msrsync -p 4 /fs/vulcan-projects/action_augment_hao/gnerv/data/DAVIS/JPEGImages/480p/ /scratch0/mgwillia/DAVIS/;"
srun bash -c "./msrsync -p 4 /fs/vulcan-projects/action_augment_hao/gnerv/data/DAVIS/Annotations/480p/ /scratch0/mgwillia/DAVIS_ANNOS/;"
srun bash -c "ls /scratch0/mgwillia;"
srun bash -c "ls /scratch0/mgwillia/DAVIS;"

srun bash -c "hostname;"
for FOLDER in ${DAVIS_FOLDERS[@]}; do
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_0;"
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_10k;"
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_20k;"
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_30k;"
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_40k;"
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_50k;"
    srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed;"
    srun bash -c "python scripts/preprocess_mask.py --annotation-path /scratch0/mgwillia/DAVIS_ANNOS/$FOLDER \
                    --mask-path /scratch0/mgwillia/DAVIS_MASKS/${FOLDER} --generate-fix-mask;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_0 \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/logs/$FOLDER/checkpoint_0.index;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_10k \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/logs/$FOLDER/checkpoint_10000.index;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_20k \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/logs/$FOLDER/checkpoint_20000.index;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_30k \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/logs/$FOLDER/checkpoint_30000.index;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_40k \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/logs/$FOLDER/checkpoint_40000.index;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/${FOLDER}_50k \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/logs/$FOLDER/checkpoint_50000.index;"
done
