#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=30Gb
#SBATCH --array=1-1334
#SBATCH --partition=xxx
#SBATCH --mail-type=all
#SBATCH --mail-user=[email address to send notification]

# This if statement below is to allow interactively test this script.
# run `sh run_jaxqtl_cis_all.sh 1` in the command line to test the file using the first line of parameter file (params_all_celltype)
if [ ! $SLURM_ARRAY_TASK_ID ]; then
    idx=$1
else
    idx=$SLURM_ARRAY_TASK_ID
fi

# TODO: change to working directory
cd xxx

# read the parameter in the ${idx}th line
params=`sed "${idx}q;d" ./data/genelist/params_all_celltype`
echo "Running instance ${idx} with params: ${params}"
set -- junk $params
shift

# parse each line
celltype=${1}
chr=${2}
chunk_file=${3}

geno="./data/geno/chr${chr}"
covar="./data/features/donor_features.tsv"
pheno="./data/pheno/${celltype}.bed.gz"
genelist="./data/genelist/${celltype}/chr${chr}/${chunk_file}"

mode="cis"
model="NB"
test_method="score"
nperm=1000
window=500000 # default
platform="cpu" # default

# prefix of result file
out="./result/cis/${celltype}/chr${chr}/${chunk_file}.${model}"

jaxqtl \
 --geno ${geno} \
 --covar ${covar} \
 --pheno ${pheno} \
 --window ${window} \
 --model ${model} \
 --mode ${mode} \
 --genelist ${genelist} \
 --test-method ${test_method} \
 --nperm ${nperm} \
 --addpc 2 \
 -p ${platform} \
 --standardize \
 --out ${out}
