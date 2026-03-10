#!/bin/bash

#SBATCH --account=bgmp
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=6
#SBATCH --job-name=whale
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --out=/gpfs/projects/bgmp/jonat/bioinfo/Bi625/ml_assignment/HappyWhaleClassifier/results/slurm/slurm-%j-%a.out
#SBATCH --err=/gpfs/projects/bgmp/jonat/bioinfo/Bi625/ml_assignment/HappyWhaleClassifier/results/slurm/slurm-%j-%a.err
#SBATCH --array=1-4

output_dir=/gpfs/projects/bgmp/jonat/bioinfo/Bi625/ml_assignment/HappyWhaleClassifier/results

#run_iter_num=$(ls -1 -d $output_dir/*/ | sed -r 's/.+_(.+)\//\1/' | sort -n | awk '{num=$1} END{print num+1}')
results_dir="${output_dir}/run_11"
mkdir -p $results_dir

# Each column represents the parameters used in a single run
epochs_config="10 10 10 10"
learn_rate_config="0.01 0.01 0.1 0.1"
batch_size_config="24 24 24 24"
optimizer_config="adam sgd adam sgd"
resolution_config="224 224 224 224"

# Get the argument for this run for each parameter
epochs=$(echo $epochs_config | sed 's/ /\n/g' | awk "NR==$SLURM_ARRAY_TASK_ID {print \$0}")
learn_rate=$(echo $learn_rate_config | sed 's/ /\n/g' | awk "NR==$SLURM_ARRAY_TASK_ID {print \$0}")
batch_size=$(echo $batch_size_config | sed 's/ /\n/g' | awk "NR==$SLURM_ARRAY_TASK_ID {print \$0}")
optimizer=$(echo $optimizer_config | sed 's/ /\n/g' | awk "NR==$SLURM_ARRAY_TASK_ID {print \$0}")
resolution=$(echo $resolution_config | sed 's/ /\n/g' | awk "NR==$SLURM_ARRAY_TASK_ID {print \$0}")

#input_file_basename=$(basename $input_file .sorted.fastq.gz)


echo "${SLURM_JOB_ID}: Multiprocessing test" > "${results_dir}/README.md"


subresults_dir="${results_dir}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
#mkdir -p $subresults_dir

mamba run -p "/projects/bgmp/shared/Bi625/ML_Assignment/Conda_Envs/HumpbackClassifierEnv" \
    /usr/bin/time -v \
        python \
            /projects/bgmp/jonat/bioinfo/Bi625/ml_assignment/HappyWhaleClassifier/dn_model.py \
                --id $SLURM_JOB_ID \
                --epochs $epochs \
                --learn-rate $learn_rate \
                --batch-size $batch_size \
                --output-dir $subresults_dir \
                --optimizer $optimizer \
                --resolution $resolution


exit