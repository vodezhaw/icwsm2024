#!/bin/bash

SCORES_FOLDER="./data/scores/"
TEST_FOLDER="./data/test/"
RESULT_FOLDER="./data/results/"
EXPERIMENTS=("compare_quantification_strategies" "out_of_domain" "prevalence_subsampling" "sample_sizes" "calibration_methods")

for exp in "${EXPERIMENTS[@]}"; do
  python -m icwsm2024.enumerate -s "${SCORES_FOLDER}" -m "${exp}" | parallel --progress "python -m icwsm2024.run -t ${TEST_FOLDER} -e {}" > "${RESULT_FOLDER}/${exp}.jsonl"
done
