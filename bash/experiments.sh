#!/bin/bash

SCORES_FOLDER="/app/scores/"
TEST_FOLDER="/app/data/test/"
RESULT_FOLDER="/app/scores/"
MODE="compare_quantification_strategies"

python -m aaai2023.enumerate -s "${SCORES_FOLDER}" -m "${MODE}" | parallel --progress "python -m aaai2023.run -t ${TEST_FOLDER} -e {}" > "${RESULT_FOLDER}/${MODE}.jsonl"
