# -*- coding: utf-8 -*-

"""
 @File    : auto_generated_data.py
 @author  : DouChangYou
 @Date    : 2023/12/25 22:25
 @Desc    : 生成项目预处理随机数据
 @license : MIT License
 @version : 1.0
 @copyright : Copyright (c) 2024-2025, DouChangYou All Rights Reserved.
"""

import os
import random
import pandas as pd


def generate_random_data(num_samples):
    """
    生成随机数据
    :param num_samples:
    :return:
    """
    # 生成数据
    data = []
    for _ in range(num_samples):
        row = {
            'gender': random.choice(['male', 'female']),
            'race/ethnicity': random.choice(['group A', 'group B', 'group C', 'group D', 'group E']),
            'parental level of education': random.choice(
                ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                 "master's degree"]),
            'lunch': random.choice(['standard', 'free/reduced']),
            'test preparation course': random.choice(['none', 'completed']),
            'math score': random.randint(0, 100),
            'reading score': random.randint(0, 100),
            'writing score': random.randint(0, 100)
        }
        data.append(row)
    # 将列表转换为 DataFrame 并返回
    df = pd.DataFrame(data)
    return df


def save_dataframe_to_csv(df, folder_path, file_name):
    """
    保存数据到 CSV 文件
    :param df:
    :param folder_path:
    :param file_name:
    :return:
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)
