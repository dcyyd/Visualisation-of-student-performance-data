"""
 @file main.py
 @author DouChangYou
 @date 2024-12-20 10:39:00
 @email dcyyd_kcug@yeah.net
 @github https://github.com/dcyyd
 @license : MIT License
 @version : 1.0
 @copyright : Copyright (c) 2024-2025, DouChangYou All Rights Reserved.
 @description 本代码主要用于对学生成绩数据进行全面的处理、分析以及可视化展示，旨在通过多种方式挖掘数据中蕴含的信息，例如不同因素对成绩的影响、成绩的分布情况等，并将分析结果和可视化图表以文件形式保存下来。
"""

import os  # 用于操作文件和目录
import sys  # 用于访问系统特定参数和函数
import logging  # 用于日志记录
import argparse  # 用于解析命令行参数
import traceback  # 用于处理异常
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和分析
import matplotlib.pyplot as plt  # 用于创建图表
from scipy.stats import normaltest  # 用于统计测试
import plotly.graph_objs as go  # 用于创建交互式图表
import plotly.io as pio  # 用于保存图表
from config import Config  # 导入自定义配置文件
from auto_generated_data import generate_random_data, save_dataframe_to_csv  # 导入自动生成数据模块


def log_config(log_file_name='运行日志.txt', encoding='gbk'):
    """
    配置日志记录器，将日志记录到指定文件中。
    :param log_file_name: 日志文件名，默认为 '运行日志.txt'
    :param encoding: 日志文件编码，默认为 'gbk'
    :return: 日志记录器对象
    """
    # 获取日志记录器实例，__name__ 通常是模块名，这里是主模块 '__main__'
    logger = logging.getLogger(__name__)
    # 设置日志记录器的总体级别为 DEBUG，这意味着所有级别（DEBUG、INFO、WARNING、ERROR、CRITICAL）的日志都会被处理
    logger.setLevel(logging.DEBUG)
    # 定义日志格式，包括时间、日志记录器名称、日志级别和日志消息
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 获取当前工作目录
    current_path = os.getcwd()
    # 拼接日志文件的完整路径
    log_file_path = os.path.join(current_path, log_file_name)
    try:
        # 创建一个文件处理器，用于将日志写入文件
        # mode='w' 表示以写入模式打开文件，如果文件存在则覆盖
        # encoding=encoding 使用指定的编码（默认为 'gbk'）
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding=encoding)
        # 设置文件处理器的日志级别为 DEBUG
        file_handler.setLevel(logging.DEBUG)
        # 为文件处理器设置之前定义的日志格式
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到日志记录器中
        logger.addHandler(file_handler)
        # 记录一条 INFO 级别的日志，表示成功配置日志记录器并创建文件处理器
        logger.info("成功配置日志记录器并创建文件处理器")
    except Exception as e:
        # 如果在配置日志记录器或创建文件处理器时发生异常
        # 记录一条 ERROR 级别的日志，包含异常信息
        logger.error(f"配置日志记录器或创建文件处理器时出错: {str(e)}")
        return None
    return logger


def handle_outliers(data, columns):
    """
    处理数据中的异常值，采用中位数和众数填补异常值。
    :param data: 包含数据的 DataFrame 对象
    :param columns: 需要处理的列名列表
    :return: 处理后的数据集和一个字典，字典中包含每列处理的异常值数量
    """
    # 初始化一个空字典，用于存储每列的异常值数量
    outlier_count = {}
    for col in columns:
        # 判断列的数据类型是否为数值类型（整数或浮点数）
        if np.issubdtype(data[col].dtype, np.number):
            # 计算第 25 百分位数（Q1）
            Q1 = data[col].quantile(0.25)
            # 计算第 75 百分位数（Q3）
            Q3 = data[col].quantile(0.75)
            # 计算四分位数间距（IQR），即 Q3 - Q1
            IQR = Q3 - Q1
            # 计算下限，即 Q1 - 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            # 计算上限，即 Q3 + 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 筛选出数据中低于下限或高于上限的异常值
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            # 计算该列的中位数
            median_value = data[col].median()
            # 将异常值替换为中位数
            data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = median_value
            # 记录该列的异常值数量
            outlier_count[col] = len(outliers)
        else:
            # 如果列不是数值类型，则计算该列每个值的出现次数
            value_counts = data[col].value_counts()
            # 获取该列的总数据量
            total_count = len(data[col])
            # 设置低频阈值，低于该频率的值将被视为异常值，这里设置为 0.01（可根据实际情况调整）
            low_frequency_threshold = 0.01
            # 筛选出出现频率低于阈值的低频值
            low_frequency_values = value_counts[value_counts / total_count < low_frequency_threshold].index
            # 筛选出数据中包含低频值的行，即异常值
            outliers = data[data[col].isin(low_frequency_values)]
            # 计算该列出现次数最多的值（众数）
            mode_value = data[col].mode()[0]
            # 将异常值替换为众数
            data.loc[data[col].isin(low_frequency_values), col] = mode_value
            # 记录该列的异常值数量
            outlier_count[col] = len(outliers)
    return data, outlier_count


def handle_missing_values(data, logger):
    """
    处理数据中的缺失值，采用中位数和众数填补缺失值。
    :param data: 包含数据的 DataFrame 对象
    :param logger: 日志记录器对象
    :return: 包含每列缺失值处理信息的列表
    """
    # 初始化一个空列表，用于存储每列缺失值的处理信息
    missing_info = []
    for col in ['数学成绩', '阅读成绩', '写作成绩']:
        # 计算该列缺失值的数量（填补前）
        before_missing_count = data[col].isnull().sum()
        # 计算该列的中位数
        median_val = data[col].median()
        # 使用中位数填充该列的缺失值
        data[col].fillna(median_val, inplace=True)
        # 计算该列缺失值的数量（填补后）
        after_missing_count = data[col].isnull().sum()
        # 生成该列缺失值处理信息的字符串
        missing_info.append(
            f"{col} 列：填补前缺失值数量 {before_missing_count}，填补后缺失值数量 {after_missing_count}，采用中位数 {median_val} 填补。")
    # 计算 '午餐' 列出现次数最多的值（众数）
    mode_lunch = data['午餐'].mode()[0]
    # 使用众数填充 '午餐' 列的缺失值
    data['午餐'].fillna(mode_lunch, inplace=True)
    # 定义一个映射字典，根据父母教育水平映射备考课程信息
    edu_level_mapping = {
        '本科及以上': '有备考课程',
        '高中': '无备考课程'
    }
    # 使用父母教育水平映射的备考课程信息填充 '备考课程' 列的缺失值
    data['备考课程'].fillna(data['父母教育水平'].map(edu_level_mapping), inplace=True)
    return missing_info


def process_data_files(config, logger, operation='append'):
    """
    处理 data_source 文件夹下的两个 CSV 文件的数据，可以选择追加或合并操作，并保存结果到一个新的 CSV 文件中。

    参数：
    - config：Config 类的实例，用于获取相关文件夹路径配置信息。
    - logger: 日志记录器对象
    - operation: 操作类型，'append' 表示追加数据，'merge' 表示合并数据，默认为 'append'

    返回值：无，将处理后的数据保存到文件。
    """
    # 记录日志，表明数据处理开始
    logger.info(f"开始 {operation} 数据文件...")

    # 获取数据源文件夹路径，通过配置对象的方法获取
    data_source_folder = os.path.join(os.getcwd(), config.get_data_source_folder())

    # 定义两个 CSV 文件的路径
    file1_path = os.path.join(data_source_folder, 'StudentsPerformance.csv')
    file2_path = os.path.join(data_source_folder, 'StudentsPerformance.Auto-generated.csv')

    try:
        # 尝试从指定路径读取第一个 CSV 数据文件
        data1 = pd.read_csv(file1_path)
        # 记录一条 INFO 级别的日志，包含成功读取的数据文件路径和数据形状
        logger.info(f"成功读取数据文件 {file1_path}，数据形状: {data1.shape}")
    except FileNotFoundError as e:
        # 如果文件不存在，记录一条 ERROR 级别的日志，包含错误信息
        logger.error(f"文件 {file1_path} 不存在，请检查文件路径！错误详情: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        # 如果文件格式有误，无法正确解析，记录一条 ERROR 级别的日志，包含错误信息
        logger.error(f"文件 {file1_path} 格式有误，无法正确解析，请核对数据格式！错误详情: {str(e)}")
        return None

    try:
        # 尝试从指定路径读取第二个 CSV 数据文件
        data2 = pd.read_csv(file2_path)
        # 记录一条 INFO 级别的日志，包含成功读取的数据文件路径和数据形状
        logger.info(f"成功读取数据文件 {file2_path}，数据形状: {data2.shape}")
    except FileNotFoundError as e:
        # 如果文件不存在，记录一条 ERROR 级别的日志，包含错误信息
        logger.error(f"文件 {file2_path} 不存在，请检查文件路径！错误详情: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        # 如果文件格式有误，无法正确解析，记录一条 ERROR 级别的日志，包含错误信息
        logger.error(f"文件 {file2_path} 格式有误，无法正确解析，请核对数据格式！错误详情: {str(e)}")
        return None

    if operation == 'append':
        # 使用 concat 函数将两个数据集进行垂直拼接
        processed_data = pd.concat([data1, data2], ignore_index=True)
        # 记录日志，表明正在进行追加操作
        logger.info("正在进行数据追加操作...")
    elif operation == 'merge':
        # 为两个数据集添加一个唯一的 student_id 列
        data1['student_id'] = range(1, len(data1) + 1)
        data2['student_id'] = range(1, len(data2) + 1)
        # 使用 'student_id' 作为连接键，将两个数据集进行左连接
        processed_data = pd.merge(data1, data2, on='student_id', how='outer', suffixes=('_original', '_generated'))
        # 记录日志，表明正在进行合并操作
        logger.info("正在进行数据合并操作...")
    else:
        # 如果操作类型不支持，记录错误日志并返回
        logger.error(f"不支持的操作类型: {operation}，请使用 'append' 或 'merge'")
        return None

    # 记录数据的初始行数
    initial_row_count = len(processed_data)
    # 使用drop_duplicates方法去除数据中的重复行，inplace=True表示直接在原数据上进行修改
    processed_data.drop_duplicates(inplace=True)
    # 计算去除重复行后的数量变化
    removed_duplicate_count = initial_row_count - len(processed_data)
    # 记录INFO级别的日志，显示去除的重复行数量
    logger.info(f"去除重复行数量: {removed_duplicate_count}")

    # 通过 Config 类获取项目运行输出文件夹路径，再拼接上处理数据输出文件的子文件夹路径
    processed_data_folder = os.path.join(os.getcwd(), config.get_project_run_output_folder(),
                                         f"{operation}数据输出文件")
    try:
        # 创建处理数据结果文件夹，如果文件夹已存在则不会报错
        os.makedirs(processed_data_folder, exist_ok=True)
        # 记录日志，表明文件夹创建成功或已存在
        logger.info(f"成功创建或已存在 {operation} 数据结果文件夹 {processed_data_folder}")
    except Exception as e:
        # 如果创建文件夹出错，记录错误日志并返回，不再继续后续分析
        logger.error(f"创建 {operation} 数据结果文件夹时出错: {str(e)}，无法继续进行数据 {operation}")
        return

    # 拼接处理后数据的输出文件路径
    output_file_path = os.path.join(processed_data_folder, f'StudentsPerformance_{operation}.csv')
    try:
        # 将处理后的数据保存为CSV文件，不保存索引，使用'utf-8-sig'编码以避免在Excel中打开时出现编码问题
        processed_data.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        # 记录INFO级别的日志，显示成功保存处理后数据的文件路径以及保存的数据行数
        logger.info(f"成功将 {operation} 后的数据保存到 {output_file_path}，保存数据行数: {processed_data.shape[0]}")
    except Exception as e:
        # 如果保存处理后的数据文件时发生异常，记录ERROR级别的日志，包含异常信息
        logger.error(f"保存 {operation} 后的数据文件时出错: {str(e)}")
        return None

    # 记录日志，表明数据合并处理完成
    logger.info(f"{operation.capitalize()} 数据完成。")


def preprocess_data(file_path, config, logger, operation='append'):
    """
    预处理数据，包括读取、备份、重命名列名、处理缺失值等。
    :param operation: 操作类型默认append
    :param file_path: 数据文件的路径
    :param config: 配置对象，提供如备份文件夹路径、项目运行输出文件夹路径、数据源文件夹路径等配置信息
    :param logger: 日志记录器对象
    :return: 预处理后的数据集
    """
    # 记录一条 INFO 级别的日志，表示开始数据预处理
    logger.info("开始数据预处理...")
    # 设置 matplotlib 绘图时使用的字体为 'Microsoft YaHei'，以正确显示中文
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 设置 matplotlib 绘图时正确显示负号
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 尝试从指定路径读取 CSV 数据文件
        data = pd.read_csv(file_path)
        # 记录一条 INFO 级别的日志，包含成功读取的数据文件路径和数据形状
        logger.info(f"成功读取数据文件 {file_path}，数据形状: {data.shape}")
    except FileNotFoundError as e:
        # 如果文件不存在，记录一条 ERROR 级别的日志，包含错误信息
        logger.error(f"文件 {file_path} 不存在，请检查文件路径！错误详情: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        # 如果文件格式有误，无法正确解析，记录一条 ERROR 级别的日志，包含错误信息
        logger.error(f"文件 {file_path} 格式有误，无法正确解析，请核对数据格式！错误详情: {str(e)}")
        return None

    # 测试process_data_files函数进行数据合并处理，实际使用时，没有用到合并的数据
    # process_data_files(config, logger, operation)

    # 获取备份文件夹路径，通过配置对象的方法获取
    backup_folder_path = os.path.join(os.getcwd(), config.get_data_backup_folder())
    # 拼接备份文件的完整路径
    backup_path = os.path.join(backup_folder_path, 'StudentsPerformance_backup.csv')
    try:
        # 创建备份文件夹，如果文件夹已存在则不报错
        os.makedirs(backup_folder_path, exist_ok=True)
        # 记录一条 INFO 级别的日志，包含成功创建或已存在的备份文件夹路径
        logger.info(f"成功创建或已存在备份文件夹 {backup_folder_path}")
        # 将原始数据保存为备份文件，不保存索引
        data.to_csv(backup_path, index=False)
        # 记录一条 INFO 级别的日志，包含成功备份的文件路径和备份数据的行数
        logger.info(f"成功备份原始数据集到 {backup_path}，备份数据行数: {data.shape[0]}")
    except Exception as e:
        # 如果备份原始数据集时发生异常，记录一条 ERROR 级别的日志，包含异常信息
        logger.error(f"备份原始数据集时出错: {str(e)}")
        return None

    # 定义列名映射字典，用于将原始列名重命名为更易读的名称
    column_mapping = {
        'gender': '性别',
        'race/ethnicity': '种族/民族',
        'parental level of education': '父母教育水平',
        'lunch': '午餐',
        'test preparation course': '备考课程',
        'math score': '数学成绩',
        'reading score': '阅读成绩',
        'writing score': '写作成绩'
    }
    # 使用列名映射字典对数据的列名进行重命名，inplace=True 表示直接在原数据上进行修改
    data.rename(columns=column_mapping, inplace=True)

    # 获取预处理输出文件夹路径，通过配置对象的方法获取
    preprocess_output_dir = os.path.join(os.getcwd(), config.get_project_run_output_folder(), "预处理输出文件")
    try:
        # 创建预处理输出文件夹，如果文件夹已存在则不报错
        os.makedirs(preprocess_output_dir, exist_ok=True)
        # 记录一条 INFO 级别的日志，包含成功创建的预处理数据文件夹路径
        logger.info(f"成功创建预处理数据文件夹 {preprocess_output_dir}")
    except Exception as e:
        # 如果创建预处理数据文件夹时发生异常，记录一条 ERROR 级别的日志，包含异常信息
        logger.error(f"创建预处理数据文件夹时出错: {str(e)}")
        return None

    # 拼接数据基本信息文件的完整路径
    info_file_path = os.path.join(preprocess_output_dir, 'data_info.txt')
    try:
        # 打开文件，以写入模式，使用 'utf-8' 编码
        with open(info_file_path, 'w', encoding='utf-8') as f:
            # 写入文件头部信息
            f.write('数据基本信息：\n')
            # 将数据的详细信息（如每列的数据类型、非空值数量等）写入文件
            data.info(buf=f)
        # 记录一条 INFO 级别的日志，包含成功保存数据基本信息的文件路径
        logger.info(f"成功将数据基本信息保存到 {info_file_path}")
    except Exception as e:
        # 如果保存数据基本信息文件时发生异常，记录一条 ERROR 级别的日志，包含异常信息
        logger.error(f"保存数据基本信息文件时出错: {str(e)}")
        return None

    # 定义需要处理异常值的所有列名列表
    all_columns = ['数学成绩', '阅读成绩', '写作成绩', '午餐', '备考课程']
    # 调用 handle_outliers 函数处理异常值，返回处理后的数据和每列的异常值数量
    data, outlier_count = handle_outliers(data, all_columns)

    # 拼接文件路径
    handle_outliers_info_file_path = os.path.join(preprocess_output_dir, 'handle_outliers_info.txt')
    try:
        # 打开文件，以写入模式，使用 'utf-8' 编码
        with open(handle_outliers_info_file_path, 'w', encoding='utf - 8') as f:
            for col, count in outlier_count.items():
                info = f"{col} 列异常值处理: 修复 {count} 个异常值\n"
                # 将每列异常值的处理信息写入文件，每行一个信息
                f.write(info)
            # 记录一条 INFO 级别的日志，包含成功保存异常值处理情况信息的文件路径
            logger.info(f"成功保存异常值处理情况信息到 {handle_outliers_info_file_path}")
    except Exception as e:
        # 如果保存异常值处理情况文件时发生异常，记录一条 ERROR 级别的日志，包含异常信息
        logger.error(f"保存异常值处理情况文件时出错: {str(e)}")
        return None

    # 调用 handle_missing_values 函数处理缺失值，返回每列缺失值的处理信息
    missing_info = handle_missing_values(data, logger)
    # 拼接缺失值处理情况信息文件的完整路径
    missing_info_file_path = os.path.join(preprocess_output_dir, 'missing_value_info.txt')
    try:
        # 打开文件，以写入模式，使用 'utf-8' 编码
        with open(missing_info_file_path, 'w', encoding='utf-8') as f:
            # 将每列缺失值的处理信息写入文件，每行一个信息
            f.write("\n".join(missing_info))
        # 记录一条 INFO 级别的日志，包含成功保存缺失值处理情况信息的文件路径
        logger.info(f"成功保存缺失值处理情况信息到 {missing_info_file_path}")
    except Exception as e:
        # 如果保存缺失值处理情况文件时发生异常，记录一条 ERROR 级别的日志，包含异常信息
        logger.error(f"保存缺失值处理情况文件时出错: {str(e)}")
        return None

    # 获取数据的行数和列数
    rows, columns = data.shape
    if rows < 100 and columns < 20:
        # 如果数据的行数小于 100 且列数小于 20
        # 拼接完整数据文件的完整路径
        full_data_file_path = os.path.join(preprocess_output_dir, 'full_data.txt')
        try:
            # 打开文件，以写入模式，使用 'utf-8' 编码
            with open(full_data_file_path, 'w', encoding='utf-8') as f:
                # 写入文件头部信息
                f.write('数据全部内容信息：\n')
                # 将数据以 markdown 格式写入文件，设置数字和字符串对齐方式
                f.write(data.to_markdown(numalign='left', stralign='left'))
            # 记录一条 INFO 级别的日志，包含成功保存全部数据内容信息的文件路径
            logger.info(f"成功保存全部数据内容信息到 {full_data_file_path}")
        except Exception as e:
            # 如果保存全部数据内容信息文件时发生异常，记录一条 ERROR 级别的日志，包含异常信息
            logger.error(f"保存全部数据内容信息文件时出错: {str(e)}")
            return None
    else:
        # 如果数据的行数大于等于 100 或列数大于等于 20
        # 拼接数据前几行内容信息文件的完整路径
        head_data_file_path = os.path.join(preprocess_output_dir, 'head_data.txt')
        try:
            # 打开文件，以写入模式，使用 'utf-8' 编码
            with open(head_data_file_path, 'w', encoding='utf-8') as f:
                # 写入文件头部信息
                f.write('数据前几行内容信息：\n')
                # 将数据的前几行以 markdown 格式写入文件，设置数字和字符串对齐方式
                f.write(data.head().to_markdown(numalign='left', stralign='left'))
            # 记录一条 INFO 级别的日志，包含成功保存数据前几行内容信息的文件路径
            logger.info(f"成功保存数据前几行内容信息到 {head_data_file_path}")
        except Exception as e:
            # 如果保存数据前几行内容信息文件时发生异常，记录一条 ERROR 级别的日志，包含异常信息
            logger.error(f"保存数据前几行内容信息文件时出错: {str(e)}")
            return None

    # 记录数据的初始行数
    initial_row_count = len(data)
    # 使用drop_duplicates方法去除数据中的重复行，inplace=True表示直接在原数据上进行修改
    data.drop_duplicates(inplace=True)
    # 计算去除重复行后的数量变化
    removed_duplicate_count = initial_row_count - len(data)
    # 记录INFO级别的日志，显示去除的重复行数量
    logger.info(f"去除重复行数量: {removed_duplicate_count}")

    # 拼接预处理后数据的输出文件路径
    output_file_path = os.path.join(preprocess_output_dir, 'StudentsPerformance_processed.csv')
    try:
        # 将处理后的数据保存为CSV文件，不保存索引，使用'utf - 8 - sig'编码以避免在Excel中打开时出现编码问题
        data.to_csv(output_file_path, index=False, encoding='utf - 8 - sig')
        # 记录INFO级别的日志，显示成功保存处理后数据的文件路径以及保存的数据行数
        logger.info(f"成功将处理后的数据保存到 {output_file_path}，保存数据行数: {data.shape[0]}")
    except Exception as e:
        # 如果保存处理后的数据文件时发生异常，记录ERROR级别的日志，包含异常信息
        logger.error(f"保存处理后的数据文件时出错: {str(e)}")
        return None
    # 记录INFO级别的日志，表示数据预处理完成
    logger.info("数据预处理完成。")
    return data


def analyze_data(data, config, logger):
    """
    函数功能：对预处理后的学生成绩数据进行分析，计算不同维度下的多种统计指标、成绩排名前 50 和倒数 50 名同学各因素影响、
            综合因素对成绩影响、各因素与成绩相关性、学生各科成绩是否符合正态分布，并将结果保存到文件。
    参数：
    - data：经过预处理后的学生成绩数据，类型为 pandas 的 DataFrame。
    - config：Config 类的实例，用于获取相关文件夹路径配置信息。
    - logger: 日志记录器对象
    返回值：无，将分析结果保存到文件。
    """
    # 记录日志，表明数据分析开始
    logger.info("开始数据分析...")

    # 定义用于分组的维度列表，这些维度将用于后续的分组统计分析
    dimensions = ['种族/民族', '性别', '父母教育水平', '午餐', '备考课程']
    # 定义成绩列名列表，这些是需要分析的成绩科目
    score_cols = ['数学成绩', '阅读成绩', '写作成绩']

    # 通过 Config 类获取项目运行输出文件夹路径，再拼接上数据分析输出文件的子文件夹路径
    analysis_results_folder = os.path.join(os.getcwd(), config.get_project_run_output_folder(), "数据分析输出文件")
    try:
        # 创建数据分析结果文件夹，如果文件夹已存在则不会报错
        os.makedirs(analysis_results_folder, exist_ok=True)
        # 记录日志，表明文件夹创建成功或已存在
        logger.info(f"成功创建或已存在数据分析结果文件夹 {analysis_results_folder}")
    except Exception as e:
        # 如果创建文件夹出错，记录错误日志并返回，不再继续后续分析
        logger.error(f"创建数据分析结果文件夹时出错: {str(e)}，无法继续进行数据分析")
        return

    # 定义一个内部函数，用于对单个维度进行统计分析并保存结果到文件，以减少代码冗余
    def analyze_dimension(dim):
        """
        针对单个维度进行统计分析，并将结果保存到对应文件。

        参数：
        dim：维度名称，字符串类型。
        """
        # 替换维度名称中的斜杠字符，避免在文件路径中出现问题
        safe_dim = dim.replace('/', '_').replace('\\', '_')
        # 定义一个字典，用于存储不同统计结果对应的文件路径
        file_paths = {
            "mean": os.path.join(analysis_results_folder, f"{safe_dim}维度平均成绩结果.txt"),
            "std": os.path.join(analysis_results_folder, f"{safe_dim}维度成绩标准差结果.txt"),
            "max": os.path.join(analysis_results_folder, f"{safe_dim}维度成绩最大值结果.txt"),
            "min": os.path.join(analysis_results_folder, f"{safe_dim}维度成绩最小值结果.txt"),
            "quantile_25": os.path.join(analysis_results_folder, f"{safe_dim}维度成绩 25 分位数结果.txt"),
            "quantile_75": os.path.join(analysis_results_folder, f"{safe_dim}维度成绩 75 分位数结果.txt"),
            "normal_test": os.path.join(analysis_results_folder, f"{safe_dim}维度成绩正态分布检验结果.txt")
        }

        # 按指定维度分组，计算每个分组中各成绩科目的平均值
        mean_scores = data.groupby(dim).mean()[score_cols]
        # 按指定维度分组，计算每个分组中各成绩科目的标准差
        std_scores = data.groupby(dim).std()[score_cols]
        # 按指定维度分组，计算每个分组中各成绩科目的最大值
        max_scores = data.groupby(dim).max()[score_cols]
        # 按指定维度分组，计算每个分组中各成绩科目的最小值
        min_scores = data.groupby(dim).min()[score_cols]
        # 按指定维度分组，计算每个分组中各成绩科目的 25 分位数
        quantile_25 = data.groupby(dim).quantile(0.25)[score_cols]
        # 按指定维度分组，计算每个分组中各成绩科目的 75 分位数
        quantile_75 = data.groupby(dim).quantile(0.75)[score_cols]

        normal_test_results = []
        for score_col in score_cols:
            # 对每个成绩科目进行正态分布检验，返回统计量和 p 值
            statistic, p_value = normaltest(data[score_col])
            # 根据 p 值判断成绩是否符合正态分布（通常 p > 0.05 认为符合正态分布）
            is_normal = p_value > 0.05
            # 将每个成绩科目的正态分布检验结果添加到列表中
            normal_test_results.append(f"{score_col}是否符合正态分布: {is_normal} (p - value: {p_value})")

        # 将各种统计结果存储在一个字典中
        results = {
            "mean": mean_scores,
            "std": std_scores,
            "max": max_scores,
            "min": min_scores,
            "quantile_25": quantile_25,
            "quantile_75": quantile_75,
            "normal_test": normal_test_results
        }

        for result_type, content in results.items():
            try:
                # 获取对应统计结果的文件路径
                file_path = file_paths[result_type]
                if result_type == "normal_test":
                    # 如果是正态分布检验结果，将列表转为字符串，每个结果占一行
                    content_str = "\n".join(content)
                else:
                    # 其他统计结果，将 DataFrame 转为字符串，并包含索引
                    content_str = content.to_string(index=True)
                # 打开文件并写入统计结果
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content_str + "\n")
                # 记录日志，表明结果写入文件成功
                logger.info(f"成功将 {result_type} 结果写入文件 {file_path}")
            except Exception as e:
                # 如果写入文件出错，记录错误日志
                logger.error(f"写入 {result_type} 结果到文件 {file_path} 时出错: {str(e)}")

    # 对每个维度调用 analyze_dimension 函数进行分析
    for dim in dimensions:
        analyze_dimension(dim)

    # 选择需要计算相关性的列，去除重复行后计算相关性矩阵
    correlation_matrix = data[
        ['性别', '父母教育水平', '午餐', '备考课程', '数学成绩', '阅读成绩', '写作成绩']].drop_duplicates().corr()
    # 构建各因素与成绩相关性结果的文件路径
    correlation_file_path = os.path.join(analysis_results_folder, "各因素与成绩相关性结果.txt")
    try:
        # 打开文件并写入相关性矩阵
        with open(correlation_file_path, 'w', encoding='utf-8') as f:
            f.write(correlation_matrix.to_string())
        # 记录日志，表明相关性结果写入文件成功
        logger.info(f"成功将各因素与成绩相关性结果写入文件 {correlation_file_path}")
        # logger.info(f"各因素与成绩相关性矩阵:\n {correlation_matrix}")
    except Exception as e:
        # 如果写入文件出错，记录错误日志
        logger.error(f"写入各因素与成绩相关性结果到文件 {correlation_file_path} 时出错: {str(e)}")

    # 计算综合因素指标，将性别、父母教育水平、午餐、备考课程等因素编码后加权求和
    data['综合因素指标'] = (data['性别'].astype('category').cat.codes * 10 +
                            data['父母教育水平'].astype('category').cat.codes * 100 +
                            data['午餐'].astype('category').cat.codes * 1000 +
                            data['备考课程'].astype('category').cat.codes * 10000)
    # 按综合因素指标分组，计算每个分组中各成绩科目的平均值
    composite_analysis = data.groupby('综合因素指标').mean()[score_cols]
    # 构建综合因素对成绩影响结果的文件路径
    composite_file_path = os.path.join(analysis_results_folder, "综合因素对成绩影响结果.txt")
    try:
        # 打开文件并写入综合因素对成绩的影响分析结果
        with open(composite_file_path, 'w', encoding='utf-8') as f:
            f.write(composite_analysis.to_string(index=True) + "\n")
        # 记录日志，表明综合因素对成绩影响结果写入文件成功
        logger.info(f"成功将综合因素对成绩影响结果写入文件 {composite_file_path}")
        # logger.info(f"综合因素对成绩的影响分析结果:\n {composite_analysis}")
    except Exception as e:
        # 如果写入文件出错，记录错误日志
        logger.error(f"写入综合因素对成绩影响结果到文件 {composite_file_path} 时出错: {str(e)}")

    # 定义一个内部函数，用于分析成绩排名前 50 或倒数 50 名同学受各因素影响情况并保存结果到文件
    def analyze_top_bottom_50(top_bottom_50_data, file_path_prefix):
        """
        分析成绩排名前 50 或倒数 50 名同学受各因素影响情况，并将相关性结果保存到文件。

        参数：
        top_bottom_50_data：成绩排名前 50 或倒数 50 名同学的数据，DataFrame 类型。
        file_path_prefix：文件路径前缀，字符串类型。
        """
        # 计算这些同学数据中各因素与成绩之间的相关性矩阵
        correlation = top_bottom_50_data[dimensions + score_cols].corr()
        # 构建对应文件路径
        file_path = os.path.join(analysis_results_folder, f"{file_path_prefix}同学各因素影响结果.txt")
        try:
            # 打开文件并写入相关性结果
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{file_path_prefix}同学各因素相关性：\n")
                f.write(correlation.to_string())
            # 记录日志，表明相关性结果写入文件成功
            logger.info(f"成功将 {file_path_prefix} 同学各因素相关性结果写入文件 {file_path}")
            # logger.info(f"{file_path_prefix}同学各因素相关性矩阵:\n {correlation}")
        except Exception as e:
            # 如果写入文件出错，记录错误日志
            logger.error(f"写入 {file_path_prefix} 同学各因素相关性结果到文件 {file_path} 时出错: {str(e)}")

    # 获取数学成绩排名前 50 的数据
    top_50_data = data.nlargest(50, '数学成绩')
    # 调用 analyze_top_bottom_50 函数分析成绩排名前 50 名同学的情况
    analyze_top_bottom_50(top_50_data, "成绩排名前 50 名")
    # 获取数学成绩排名倒数 50 的数据
    bottom_50_data = data.nsmallest(50, '数学成绩')
    # 调用 analyze_top_bottom_50 函数分析成绩排名倒数 50 名同学的情况
    analyze_top_bottom_50(bottom_50_data, "成绩排名倒数 50 名")

    # 记录日志，表明数据分析完成
    logger.info("数据分析完成。")


def visualize_data(data, config, logger):
    """
    对给定的学生成绩数据进行可视化展示，并将生成的可视化图表保存到指定的输出文件夹中。

    参数：
    - data：经过预处理后的学生成绩数据，类型为 pandas 的 DataFrame。
    - config：Config 类的实例，用于获取相关文件夹路径配置信息。
    - logger：日志记录器对象

    返回值：无
    """
    # 记录日志，表明数据可视化开始
    logger.info("开始数据可视化...")

    # 获取可视化结果的基础文件夹路径，通过拼接当前工作目录、项目运行输出文件夹和可视化输出文件子文件夹得到
    visualization_base_folder = os.path.join(os.getcwd(), config.get_project_run_output_folder(), "可视化输出文件")

    # 定义一个字典，用于存储不同类型可视化图表对应的文件夹路径和名称
    chart_dirs = {
        "bar_chart": os.path.join(visualization_base_folder, "柱状图"),
        "line_chart": os.path.join(visualization_base_folder, "折线图"),
        "pie_chart": os.path.join(visualization_base_folder, "饼图"),
        "box_chart": os.path.join(visualization_base_folder, "箱图")
    }

    # 遍历 chart_dirs 字典，创建每个类型可视化图表存放的文件夹，并添加错误处理和日志记录
    for dir_name, dir_path in chart_dirs.items():
        try:
            # 创建文件夹，如果文件夹已存在则不会引发异常
            os.makedirs(dir_path, exist_ok=True)
            # 记录日志，表明文件夹创建成功
            logger.info(f"成功创建 {dir_name} 图表存放文件夹 {dir_path}")
        except Exception as e:
            # 如果创建文件夹出错，记录错误日志，然后继续尝试创建下一个文件夹
            logger.error(f"创建 {dir_name} 图表存放文件夹 {dir_path} 时出错: {str(e)}")
            continue

    # 定义各因素维度列表，这些因素将用于后续可视化分析
    dimensions = ['父母教育水平', '性别', '种族/民族', '午餐', '备考课程']
    # 定义成绩科目列名列表，这些成绩科目将用于可视化展示
    score_cols = ['数学成绩', '阅读成绩', '写作成绩']

    # 定义一个函数，用于绘制并保存柱状图，以减少代码冗余
    def draw_and_save_bar_charts():
        # 遍历每个因素维度
        for factor in dimensions:
            # 创建一个 Plotly 的 Figure 对象，用于绘制图表
            fig = go.Figure()
            # 针对每个成绩科目
            for score_col in score_cols:
                # 按因素分组，计算每个分组中成绩科目的平均值，并重置索引，得到用于绘图的数据
                grouped_data = data.groupby(factor)[score_col].mean().reset_index()
                # 在图表中添加一个柱状图轨迹，x 轴为因素分组，y 轴为该成绩科目的平均成绩，轨迹名称为成绩科目名称
                fig.add_trace(go.Bar(x=grouped_data[factor], y=grouped_data[score_col], name=score_col))
            # 更新图表布局，设置图表标题为该因素对学生各科成绩的综合影响，x 轴标题为因素名称，y 轴标题为成绩
            fig.update_layout(title=f'{factor}对学生各科成绩的综合影响', xaxis_title=factor, yaxis_title='成绩')
            # 替换因素名称中的斜杠字符，避免在文件路径中出现问题
            safe_factor = factor.replace('/', '_').replace('\\', '_')
            # 构建保存柱状图的文件路径，路径位于柱状图文件夹内，文件名包含因素名称
            file_path = os.path.join(chart_dirs["bar_chart"], f'{safe_factor}对学生各科成绩综合影响柱状图.html')
            try:
                # 使用 Plotly 的 write_html 方法将图表保存为 HTML 文件
                pio.write_html(fig, file=file_path)
                # 记录日志，表明柱状图保存成功
                logger.info(f"成功保存 {factor} 对学生各科成绩综合影响柱状图到 {file_path}")
            except Exception as e:
                # 如果保存文件出错，记录错误日志
                logger.error(f"保存 {factor} 对学生各科成绩综合影响柱状图到 {file_path} 时出错: {str(e)}")

    # 定义一个函数，用于绘制并保存折线图，以减少代码冗余
    def draw_and_save_line_charts():
        # 遍历每个成绩科目
        for score_col in score_cols:
            # 创建一个 Plotly 的 Figure 对象，用于绘制图表
            fig = go.Figure()
            # 针对每个因素维度
            for dim in dimensions:
                # 按维度分组，计算每个分组中成绩科目的平均值，并重置索引，得到用于绘图的数据
                grouped_data = data.groupby(dim)[score_col].mean().reset_index()
                # 在图表中添加一个折线图轨迹，x 轴为维度分组，y 轴为该成绩科目的平均成绩，轨迹名称为维度名称
                fig.add_trace(
                    go.Scatter(x=grouped_data[dim], y=grouped_data[score_col], mode='lines+markers', name=dim))
            # 更新图表布局，设置图表标题为各因素对该成绩科目的影响综合对比，x 轴标题为影响因素，y 轴标题为成绩科目名称
            fig.update_layout(title=f'各因素对{score_col}的影响综合对比', xaxis_title='影响因素',
                              yaxis_title=score_col)
            # 替换成绩科目名称中的斜杠字符，避免在文件路径中出现问题
            safe_factor = score_col.replace('/', '_').replace('\\', '_')
            # 构建保存折线图的文件路径，路径位于折线图文件夹内，文件名包含成绩科目名称
            file_path = os.path.join(chart_dirs["line_chart"], f"各因素对{safe_factor}影响综合对比折线图.html")
            try:
                # 使用 Plotly 的 write_html 方法将图表保存为 HTML 文件
                pio.write_html(fig, file=file_path)
                # 记录日志，表明折线图保存成功
                logger.info(f"成功保存各因素对 {score_col} 影响综合对比折线图到 {file_path}")
            except Exception as e:
                # 如果保存文件出错，记录错误日志
                logger.error(f"保存各因素对 {score_col} 影响综合对比折线图到 {file_path} 时出错: {str(e)}")

    # 定义一个函数，用于绘制并保存饼图，以减少代码冗余
    def draw_and_save_pie_charts():
        # 定义一个内部函数，用于生成单个成绩科目的等级分布及及格率饼图
        def generate_dynamic_score_pie_chart(score_col):
            # 定义成绩等级标签列表
            labels = ['不及格', '及格', '中', '良', '优']
            # 定义成绩等级的区间边界列表
            bins = [0, 60, 70, 80, 90, 100]
            # 将成绩按照等级区间进行分组，并统计每个等级的数量，然后按索引排序
            sizes = pd.cut(data[score_col], bins=bins, labels=labels).value_counts().sort_index()
            # 计算及格率，即及格及以上等级的数量之和占总数量的百分比
            pass_rate = (sizes.loc[['及格', '中', '良', '优']].sum() / sizes.sum()) * 100
            # 创建一个 Plotly 的饼图 Figure 对象，数据为成绩等级标签和对应的数量
            fig = go.Figure(data=[go.Pie(labels=labels, values=sizes.values)])
            # 更新图表布局，设置图表标题为该成绩科目的等级分布及及格率，并显示及格率数值
            fig.update_layout(title=f'{score_col}等级分布及及格率（及格率：{pass_rate:.2f}%）')
            # 构建保存饼图的文件路径，路径位于饼图文件夹内，文件名包含成绩科目名称
            file_path = os.path.join(chart_dirs["pie_chart"], f"{score_col}等级分布及及格率动态饼图.html")
            try:
                # 使用 Plotly 的 write_html 方法将图表保存为 HTML 文件
                pio.write_html(fig, file=file_path)
                # 记录日志，表明饼图保存成功
                logger.info(f"成功保存 {score_col} 等级分布及及格率动态饼图到 {file_path}")
            except Exception as e:
                # 如果保存文件出错，记录错误日志
                logger.error(f"保存 {score_col} 等级分布及及格率动态饼图到 {file_path} 时出错: {str(e)}")

        # 对每个成绩科目调用 generate_dynamic_score_pie_chart 函数生成并保存饼图
        for score_col in score_cols:
            generate_dynamic_score_pie_chart(score_col)

    # 定义一个函数，用于绘制并保存箱图，以减少代码冗余
    def draw_and_save_box_charts():
        # 创建一个 Plotly 的 Figure 对象，用于绘制图表
        fig = go.Figure()
        # 遍历每个因素维度
        for col_idx, dim in enumerate(dimensions):
            # 针对每个成绩科目
            for row_idx, score_col in enumerate(score_cols):
                # 在图表中添加一个箱图轨迹，x 轴为因素维度，y 轴为成绩科目，轨迹名称为因素维度 - 成绩科目
                fig.add_trace(go.Box(x=data[dim], y=data[score_col], name=f'{dim} - {score_col}'))
        # 更新图表布局，设置图表标题为多维度多科目成绩箱型图
        fig.update_layout(title='多维度多科目成绩箱型图')
        # 构建保存箱图的文件路径，路径位于箱图文件夹内
        file_path = os.path.join(chart_dirs["box_chart"], "多维度多科目成绩箱型图.html")
        try:
            # 使用 Plotly 的 write_html 方法将图表保存为 HTML 文件
            pio.write_html(fig, file=file_path)
            # 记录日志，表明箱图保存成功
            logger.info(f"成功保存多维度多科目成绩箱型图到 {file_path}")
        except Exception as e:
            # 如果保存文件出错，记录错误日志
            logger.error(f"保存多维度多科目成绩箱型图到 {file_path} 时出错: {str(e)}")

    # 调用绘制并保存柱状图的函数
    draw_and_save_bar_charts()
    # 调用绘制并保存折线图的函数
    draw_and_save_line_charts()
    # 调用绘制并保存饼图的函数
    draw_and_save_pie_charts()
    # 调用绘制并保存箱图的函数
    draw_and_save_box_charts()

    # 记录日志，表明数据可视化完成
    logger.info("数据可视化完成。")


def main():
    # 创建 Config 类的实例，用于获取配置信息
    config = Config()
    # 获取日志记录器对象
    logger = log_config()
    if logger is None:
        # 如果日志记录器创建失败，直接返回，不再继续执行后续代码
        return
    # 记录日志，欢迎使用学生成绩分析系统
    logger.info("欢迎使用学生成绩分析系统！")
    # 记录日志，表明程序开始运行
    logger.info("程序开始运行...")

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="学生成绩分析系统")
    # 添加样本数量参数，用户可以通过命令行指定样本数量，默认值为 1000
    parser.add_argument('--num_samples', type=int, default=1000, help='生成的随机数据样本数量，默认为 1000')
    # 解析命令行参数
    args = parser.parse_args()

    # 使用解析后的样本数量更新 Config 实例中的样本数量
    config.set_num_samples(args.num_samples)

    # 构建数据文件的路径，通过拼接当前工作目录、数据来源文件夹和具体文件名得到
    data_file_path = os.path.join(os.getcwd(), config.get_data_source_folder(), 'StudentsPerformance.csv')
    auto_generated_data_file_path = os.path.join(os.getcwd(), config.get_data_source_folder(),
                                                 'StudentsPerformance.Auto-generated.csv')

    # 检查数据文件是否存在
    if not os.path.exists(data_file_path):
        # 如果数据文件不存在，记录日志并生成随机数据
        logger.warning("数据文件不存在，将生成随机数据...")
        # 获取配置中的样本数量
        num_samples = config.get_num_samples()
        # 添加调试信息，检查 num_samples 的值
        logger.debug(f"配置中的样本数量: {num_samples}")
        if num_samples is None:
            logger.error("配置中的样本数量未正确设置，无法生成随机数据。")
            return
        # 调用生成随机数据的函数，传入数据量，获取随机数据
        df = generate_random_data(num_samples)
        # 添加调试信息，检查 df 的类型
        logger.debug(f"生成的数据类型: {type(df)}")
        # 调用保存数据到CSV文件的函数，传入数据、数据文件保存路径和文件名
        save_dataframe_to_csv(df, config.get_data_source_folder(), 'StudentsPerformance.Auto-generated.csv')
        logger.info("随机数据生成并保存完成。")
        # 更新 data_file_path 为自动生成的数据文件路径
        data_file_path = auto_generated_data_file_path
    else:
        # 如果数据文件存在，记录日志
        logger.info("数据文件已存在，将使用现有数据文件进行分析。")

    error_occurred = False  # 添加一个标志变量来记录是否发生错误

    try:
        # 调用数据预处理函数，传入文件路径、配置实例和日志记录器，获取预处理后的数据
        processed_data = preprocess_data(data_file_path, config, logger)
        if processed_data is not None:
            # 如果数据预处理成功，记录日志，并继续进行数据分析和可视化
            logger.info("数据预处理成功，继续进行数据分析和可视化。")
            # 调用数据分析函数，传入预处理后的数据、配置实例和日志记录器
            analyze_data(processed_data, config, logger)  # 确保这个函数已经定义
            # 调用数据可视化函数，传入预处理后的数据、配置实例和日志记录器
            visualize_data(processed_data, config, logger)
        else:
            # 如果数据预处理失败，记录错误日志，并设置错误标志为True
            logger.error("数据预处理失败，无法继续进行数据分析和可视化。")
            error_occurred = True
    except Exception as e:
        # 记录错误日志和堆栈跟踪信息，并设置错误标志为True
        print("程序运行出错，请查看运行日志！！！")
        logger.error("程序运行过程中出现未捕获的异常，错误信息如下：")
        logger.error(''.join(traceback.format_exception(*sys.exc_info())))
        error_occurred = True
    finally:
        # 根据是否发生错误来记录不同的日志信息
        if not error_occurred:
            logger.info("程序运行完成。")


if __name__ == '__main__':
    main()
