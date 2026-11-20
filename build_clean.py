# -*- coding: utf-8 -*-
"""
build_clean.py — 重建 PUC 清洗数据集与划分行索引(仓库不分发 Kaggle 原始/清洗记录)
================================================================
依据论文 Data Availability:"we do not redistribute its raw records"。
本脚本让你从自行下载的 Kaggle 原始文件重建分析所用的清洗表,并输出
种子 42 的 7:3 分层划分行索引(仅整数索引,可公开,用于核对划分一致性)。

用法:
  1. 从 Kaggle 下载数据集 pkdarabi/diabetes-dataset-with-18-features
     (https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features),
     得到 4,303 行的 diabetes.csv,放到本目录;
     或安装 kagglehub 后运行: kagglehub.dataset_download("pkdarabi/diabetes-dataset-with-18-features")
  2. python build_clean.py

产出:
  chinese_clean.csv        4,061 行清洗表(与 2_中国数据建模.py 同一函数、同一代码路径)
                           —— 本地生成,不入库(.gitignore 已排除)
  split_indices_seed42.csv 4,061 行的 clean_row/set(train/test)划分索引 —— 可入库

校验(与 README 期望值比对,不一致说明下载文件或环境不同):
  原始 diabetes.csv   4,303 行  sha256 = cced15f9f4aa0cd17089a2508e321f6a556e8a08b049404c5502c80fd9dd1351
  清洗 chinese_clean  4,061 行  sha256 = aff50167d373744c64735bd270292f28b890a3579e80b17b039d45d484dccdfd
  阳性 1,155 (28.4%);划分 训练 2,842 / 测试 1,219
"""
import hashlib
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from 补充实验工具 import qc_puc, SEED

RAW = 'diabetes.csv'
CLEAN = 'chinese_clean.csv'
IDX = 'split_indices_seed42.csv'
EXPECT_RAW = ('4303', 'cced15f9f4aa0cd17089a2508e321f6a556e8a08b049404c5502c80fd9dd1351')
EXPECT_CLEAN = ('4061', 'aff50167d373744c64735bd270292f28b890a3579e80b17b039d45d484dccdfd')


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    if not os.path.exists(RAW):
        print('未找到 %s。请先从 Kaggle 下载 pkdarabi/diabetes-dataset-with-18-features：' % RAW)
        print('  https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features')
        print('或将文件放到本目录后重试（也可 pip install kagglehub 后用 '
              'kagglehub.dataset_download("pkdarabi/diabetes-dataset-with-18-features") 获取）。')
        sys.exit(1)

    n_raw, h_raw = len(pd.read_csv(RAW)), sha256(RAW)
    print('[1/3] 原始文件 %s: %s 行  sha256=%s' % (RAW, n_raw, h_raw))
    if (str(n_raw), h_raw) != EXPECT_RAW:
        print('  [警告] 与期望值 %s 行 / %s 不一致——请确认下载的是同一版本数据'
              % EXPECT_RAW)

    # 与 2_中国数据建模.py 完全同一 QC 代码路径（补充实验工具.qc_puc）
    df = qc_puc()
    print('[2/3] 质控后 %d 行（期望 4061），阳性 %d（%.1f%%）'
          % (len(df), int(df['Diabetes'].sum()), df['Diabetes'].mean() * 100))
    assert len(df) == 4061, '质控后行数与主分析不一致，请检查原始文件'
    df.to_csv(CLEAN, index=False, encoding='utf-8-sig')   # 与 2 号脚本同一写出方式
    n_c, h_c = len(df), sha256(CLEAN)
    if (str(n_c), h_c) == EXPECT_CLEAN:
        print('  chinese_clean.csv sha256 与主分析逐位一致 ✓')
    else:
        print('  [警告] chinese_clean.csv sha256=%s 与期望 %s 不一致（环境差异？）'
              % (h_c, EXPECT_CLEAN[1]))

    # 种子 42、7:3 分层划分的行索引（与 2_中国数据建模.py 的 train_test_split 同参）
    tr, te = train_test_split(range(len(df)), test_size=0.3,
                              random_state=SEED, stratify=df['Diabetes'])
    tr_set = set(tr)
    idx = pd.DataFrame({'clean_row': range(len(df)),
                        'set': ['train' if i in tr_set else 'test'
                                for i in range(len(df))]})
    idx.to_csv(IDX, index=False, encoding='utf-8-sig')
    print('[3/3] 已写 %s（train %d / test %d，期望 2842/1219）'
          % (IDX, len(tr), len(te)))
    print('完成。%s 为本地重建文件（.gitignore 已排除，勿提交）；%s 仅含整数索引，可随仓库分发。'
          % (CLEAN, IDX))


if __name__ == '__main__':
    main()
