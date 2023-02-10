#!/bin/bash

# コマンドの表示・停止など
set -eux

# 実行位置に移動
SCRIPT_PATH=$0
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
cd ${SCRIPT_DIR}/../python/

# libtorchを手動導入している関係でLD_LIBRARY_PATHを
# クリアしないとpythonでpytorchが上手く動かない
export LD_LIBRARY_PATH=

# データセット準備
python make_dataset.py

# 学習
python train.py

# 推論
python infer.py
