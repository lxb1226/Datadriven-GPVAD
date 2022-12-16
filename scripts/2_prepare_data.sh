#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"




base_dir=${1-"../../data"}
output_label_dir=${2-"../../data/labels"}
output_hdf5_dir=${3-"../../data/hdf5"}

# 提取特征
# 生成标签

