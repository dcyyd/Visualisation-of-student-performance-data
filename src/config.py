# -*- coding: utf-8 -*-

"""
 @File    : config.py
 @author  : DouChangYou
 @Date    : 2024/12/20 15:25
 @Desc    : 配置类，用于管理数据源文件夹路径、备份文件夹路径和分析结果输出文件夹路径。
 @license : MIT License
 @version : 1.0
 @copyright : Copyright (c) 2024-2025, DouChangYou All Rights Reserved.
"""

class Config:
    """
    配置类，用于管理数据源文件夹路径、备份文件夹路径和分析结果输出文件夹路径。
    """

    def __init__(self, num_samples=1000):  # 添加 num_samples 参数，默认值为 1000
        self._data_source_folder = 'data_source'
        self._data_backup_folder = 'data_backup'
        self._project_run_output_folder = 'AnalyticalResults'
        self._num_samples = num_samples  # 初始化样本数量

    # Getter 和 Setter 方法
    def get_data_source_folder(self):
        return self._data_source_folder

    def set_data_source_folder(self, value):
        self._data_source_folder = value

    def get_data_backup_folder(self):
        return self._data_backup_folder

    def set_data_backup_folder(self, value):
        self._data_backup_folder = value

    def get_project_run_output_folder(self):
        return self._project_run_output_folder

    def set_project_run_output_folder(self, value):
        self._project_run_output_folder = value

    def get_num_samples(self):
        return self._num_samples  # 返回样本数量

    def set_num_samples(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("样本数量必须是正整数")
        self._num_samples = value  # 设置样本数量
