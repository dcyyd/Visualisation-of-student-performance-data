"""
- :date : 2024-12-20
- :author : DouChangYou
- :blog : https://kfufys.top
- :email : dcyyd_kcug@yeah.net
- :github : https://github.com/dcyyd
- :version : 1.0
- :license : MIT
- :copyright : Copyright (c) 2024 DouChangYou
- :description : 本代码主要用于对学生成绩数据进行全面的处理、分析以及可视化展示，旨在通过多种方式挖掘数据中蕴含的信息，例如不同因素对成绩的影响、成绩的分布情况等，并将分析结果和可视化图表以文件形式保存下来。
"""

import os  # 用于操作文件和目录
import mpld3  # 用于将matplotlib图表转换为HTML5
import pandas as pd  # 用于数据处理和分析
import seaborn as sns  # 基于matplotlib的高级绘图库
import matplotlib as mpl  # 绘图库
import matplotlib.pyplot as plt  # 用于创建图表
import plotly.express as px  # 用于创建动态可视化图表
from scipy.stats import stats, normaltest  # 用于统计测试

# 定义全局变量，指定数据源文件夹、备份文件夹和分析结果输出文件夹的路径
_src_folder = 'data_source'
_dst_folder = 'data_backup'
_analytical_results_output_folder = './AnalyticalResults'


# 数据预处理函数
def preprocess_data(file_path, output_dir):
    """
    对给定路径的学生成绩数据文件进行预处理操作，包括数据清洗、合并操作，并备份原始数据集。

    参数：
    - file_path：数据文件的路径，类型为字符串，指向包含学生成绩数据的CSV文件。
    - output_dir：输出文件夹路径，类型为字符串，用于保存处理后的数据文件以及后续生成的可视化相关文件。

    返回值：
    - 经过预处理后的学生成绩数据，类型为pandas的DataFrame。
    """

    # 添加中文字体支持，以确保图表中可以显示中文
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    mpl.rcParams['axes.unicode_minus'] = False

    # 尝试读取CSV文件，并添加错误处理机制
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，请检查文件路径！")
        return
    except pd.errors.ParserError:
        print(f"文件 {file_path} 格式有误，无法正确解析，请核对数据格式！")
        return

    # 备份原始数据集到指定的备份文件夹
    backup_path = os.path.join(_dst_folder, 'StudentsPerformance_backup.csv')
    os.makedirs(_dst_folder, exist_ok=True)  # 确保备份文件夹存在
    data.to_csv(backup_path, index=False)

    # 将列名换成中文，通过映射字典方式实现，更灵活方便维护
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
    data.rename(columns=column_mapping, inplace=True)

    # 创建预处理数据文件夹，用于保存预处理后的数据文件
    preprocess_output_dir = os.path.join(output_dir, "预处理数据")
    os.makedirs(preprocess_output_dir, exist_ok=True)

    # 将数据基本信息保存到文件
    info_file_path = os.path.join(preprocess_output_dir, 'data_info.txt')
    with open(info_file_path, 'w', encoding='utf-8') as f:
        f.write('数据基本信息：\n')
        data.info(buf=f)

    # 查看数据集行数和列数，如果数据集较小，则保存全部数据内容信息
    rows, columns = data.shape
    if rows < 100 and columns < 20:
        full_data_file_path = os.path.join(preprocess_output_dir, 'full_data.txt')
        with open(full_data_file_path, 'w', encoding='utf-8') as f:
            f.write('数据全部内容信息：\n')
            f.write(data.to_markdown(numalign='left', stralign='left'))
    else:
        # 如果数据集较大，则只保存前几行数据内容信息
        head_data_file_path = os.path.join(preprocess_output_dir, 'head_data.txt')
        with open(head_data_file_path, 'w', encoding='utf-8') as f:
            f.write('数据前几行内容信息：\n')
            f.write(data.head().to_markdown(numalign='left', stralign='left'))

    # 处理缺失值，记录处理情况到文件
    missing_info = []
    for col in ['数学成绩', '阅读成绩', '写作成绩']:
        before_missing_count = data[col].isnull().sum()
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)
        after_missing_count = data[col].isnull().sum()
        missing_info.append(
            f"{col} 列：填补前缺失值数量 {before_missing_count}，填补后缺失值数量 {after_missing_count}，采用中位数 {median_val} 填补。")
    mode_lunch = data['午餐'].mode()[0]
    data['午餐'].fillna(mode_lunch, inplace=True)
    edu_level_mapping = {
        '本科及以上': '有备考课程',
        '高中': '无备考课程'
    }
    data['备考课程'].fillna(data['父母教育水平'].map(edu_level_mapping), inplace=True)
    with open(os.path.join(preprocess_output_dir, 'missing_value_info.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(missing_info))

    # 尝试读取补充数据文件，并与主数据集合并
    supplementary_file_path = os.path.join(_src_folder, 'supplementary_data.csv')
    if os.path.exists(supplementary_file_path):
        supplementary_data = pd.read_csv(supplementary_file_path)
        data = pd.merge(data, supplementary_data, on='学生编号', how='left')

    # 数据清洗：去除重复行
    data.drop_duplicates(inplace=True)

    # 将处理后的数据保存到输出文件夹下指定的文件中
    output_file_path = os.path.join(preprocess_output_dir, 'StudentsPerformance_processed.csv')
    data.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    return data


def analyze_data(data, output_dir):
    """
    函数功能：对预处理后的学生成绩数据进行分析，计算不同维度下的多种统计指标、成绩排名前50和倒数50名同学各因素影响、
              综合因素对成绩影响、各因素与成绩相关性、学生各科成绩是否符合正态分布，并将结果保存到文件。
    参数：
    data：经过预处理后的学生成绩数据，类型为pandas的DataFrame。
    output_dir：输出文件夹路径，类型为字符串。
    返回值：无，将分析结果保存到文件。
    """
    # 定义用于分组的维度列表，包括种族/民族、性别、父母教育水平、午餐、备考课程
    dimensions = ['种族/民族', '性别', '父母教育水平', '午餐', '备考课程']
    # 定义成绩列名列表，包括数学成绩、阅读成绩、写作成绩
    score_cols = ['数学成绩', '阅读成绩', '写作成绩']

    # 确保output目录存在
    analyze_output_dir = os.path.join(output_dir, "数据分析结果")
    os.makedirs(analyze_output_dir, exist_ok=True)

    # 对每个维度进行统计分析
    for dim in dimensions:
        # 计算该维度下各科成绩的平均值
        mean_scores = data.groupby(dim).mean()[score_cols]
        # 计算该维度下各科成绩的标准差
        std_scores = data.groupby(dim).std()[score_cols]
        # 计算该维度下各科成绩的最大值
        max_scores = data.groupby(dim).max()[score_cols]
        # 计算该维度下各科成绩的最小值
        min_scores = data.groupby(dim).min()[score_cols]
        # 计算该维度下各科成绩的25分位数
        quantile_25 = data.groupby(dim).quantile(0.25)[score_cols]
        # 计算该维度下各科成绩的75分位数
        quantile_75 = data.groupby(dim).quantile(0.75)[score_cols]

        # 检查各科成绩是否符合正态分布
        normal_test_results = []
        for score_col in score_cols:
            # 使用normaltest函数进行正态性检验，返回统计量和p值
            statistic, p_value = normaltest(data[score_col])
            # 判断是否符合正态分布（通常p值大于0.05认为符合正态分布）
            is_normal = p_value > 0.05
            # 将结果添加到列表中，格式为"成绩列名是否符合正态分布: True/False (p - value: 具体p值)"
            normal_test_results.append(f"{score_col}是否符合正态分布: {is_normal} (p - value: {p_value})")

        # 使用_替换维度名中的/和\\，防止路径错误
        safe_dim = dim.replace('/', '_').replace('\\', '_')
        # 构建保存该维度平均成绩结果的文件路径
        mean_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度平均成绩结果.txt")
        # 构建保存该维度成绩标准差结果的文件路径
        std_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度成绩标准差结果.txt")
        # 构建保存该维度成绩最大值结果的文件路径
        max_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度成绩最大值结果.txt")
        # 构建保存该维度成绩最小值结果的文件路径
        min_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度成绩最小值结果.txt")
        # 构建保存该维度成绩25分位数结果的文件路径
        quantile_25_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度成绩25分位数结果.txt")
        # 构建保存该维度成绩75分位数结果的文件路径
        quantile_75_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度成绩75分位数结果.txt")
        # 构建保存该维度成绩正态分布检验结果的文件路径
        normal_test_file_path = os.path.join(analyze_output_dir, f"{safe_dim}维度成绩正态分布检验结果.txt")

        # 将平均成绩结果写入文件
        with open(mean_file_path, 'w', encoding='utf-8') as f_mean:
            f_mean.write(mean_scores.to_string(index=True) + "\n")
        # 将标准差结果写入文件
        with open(std_file_path, 'w', encoding='utf-8') as f_std:
            f_std.write(std_scores.to_string(index=True) + "\n")
        # 将最大值结果写入文件
        with open(max_file_path, 'w', encoding='utf-8') as f_max:
            f_max.write(max_scores.to_string(index=True) + "\n")
        # 将最小值结果写入文件
        with open(min_file_path, 'w', encoding='utf-8') as f_min:
            f_min.write(min_scores.to_string(index=True) + "\n")
        # 将25分位数结果写入文件
        with open(quantile_25_file_path, 'w', encoding='utf-8') as f_quantile_25:
            f_quantile_25.write(quantile_25.to_string(index=True) + "\n")
        # 将75分位数结果写入文件
        with open(quantile_75_file_path, 'w', encoding='utf-8') as f_quantile_75:
            f_quantile_75.write(quantile_75.to_string(index=True) + "\n")
        # 将正态分布检验结果写入文件
        with open(normal_test_file_path, 'w', encoding='utf-8') as f_normal_test:
            f_normal_test.write("\n".join(normal_test_results))

    # 计算各因素与成绩之间的相关性系数
    # 从数据中选择性别、父母教育水平、午餐、备考课程、数学成绩、阅读成绩、写作成绩列，去除重复行后计算相关性矩阵
    correlation_matrix = data[
        ['性别', '父母教育水平', '午餐', '备考课程', '数学成绩', '阅读成绩', '写作成绩']].drop_duplicates().corr()
    # 构建保存各因素与成绩相关性结果的文件路径
    correlation_file_path = os.path.join(analyze_output_dir, "各因素与成绩相关性结果.txt")
    with open(correlation_file_path, 'w', encoding='utf-8') as f:
        # 将相关性矩阵结果写入文件
        f.write(correlation_matrix.to_string())

    # 计算综合因素对成绩的影响
    # 通过将性别、父母教育水平、午餐、备考课程进行编码后组合成一个综合因素指标
    data['综合因素指标'] = (data['性别'].astype('category').cat.codes * 10 +
                            data['父母教育水平'].astype('category').cat.codes * 100 +
                            data['午餐'].astype('category').cat.codes * 1000 +
                            data['备考课程'].astype('category').cat.codes * 10000)
    # 按综合因素指标分组，计算各科成绩的平均值
    composite_analysis = data.groupby('综合因素指标').mean()[score_cols]
    # 构建保存综合因素对成绩影响结果的文件路径
    composite_file_path = os.path.join(analyze_output_dir, "综合因素对成绩影响结果.txt")
    with open(composite_file_path, 'w', encoding='utf-8') as f:
        # 将综合因素对成绩影响结果写入文件
        f.write(composite_analysis.to_string(index=True) + "\n")

    # 分析成绩排名前50名同学受各因素影响最大的情况
    # 选取数学成绩最高的50名同学的数据
    top_50_data = data.nlargest(50, '数学成绩')
    # 计算这50名同学的各因素与各科成绩的相关性
    top_50_correlation = top_50_data[dimensions + score_cols].corr()
    # 构建保存成绩排名前50名同学各因素影响结果的文件路径
    top_50_influence_file_path = os.path.join(analyze_output_dir, "成绩排名前50名同学各因素影响结果.txt")
    with open(top_50_influence_file_path, 'w', encoding='utf-8') as f:
        # 将成绩排名前50名同学各因素相关性结果写入文件
        f.write("成绩排名前50名同学各因素相关性：\n")
        f.write(top_50_correlation.to_string())

    # 分析成绩排名倒数50名同学受各因素影响最大的情况
    # 选取数学成绩最低的50名同学的数据
    bottom_50_data = data.nsmallest(50, '数学成绩')
    # 计算这50名同学的各因素与各科成绩的相关性
    bottom_50_correlation = bottom_50_data[dimensions + score_cols].corr()
    # 构建保存成绩排名倒数50名同学各因素影响结果的文件路径
    bottom_50_influence_file_path = os.path.join(analyze_output_dir, "成绩排名倒数50名同学各因素影响结果.txt")
    with open(bottom_50_influence_file_path, 'w', encoding='utf-8') as f:
        # 将成绩排名倒数50名同学各因素相关性结果写入文件
        f.write("成绩排名倒数50名同学各因素相关性：\n")
        f.write(bottom_50_correlation.to_string())


# 数据可视化函数
def visualize_data(data, output_dir):
    """
    函数功能：对学生成绩进行可视化展示，通过柱状图展示成绩排名前50名以及倒数50名同学受各因素影响情况，
              同时保留原有的如各因素对各科成绩影响、成绩等级分布、及格率等相关可视化内容，并将这些可视化图表转换为 HTML 文档保存到指定的输出文件夹中，
              对每种可视化结果进行简单分析论述。

    参数：
    - data：经过预处理后的学生成绩数据，类型为 pandas 的 DataFrame。
    - output_dir：输出文件夹路径，类型为字符串，用于保存生成的可视化相关的 HTML 文件。

    返回值：无，直接将可视化图表保存为 HTML 文件到指定输出文件夹中。
    """
    # 设置统一的图表风格，使图表更美观协调
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # 创建不同类型可视化图像存放的文件夹
    bar_chart_dir = os.path.join(output_dir, "可视化结果", "柱状图")
    line_chart_dir = os.path.join(output_dir, "可视化结果", "折线图")
    pie_chart_dir = os.path.join(output_dir, "可视化结果", "饼图")
    box_chart_dir = os.path.join(output_dir, "可视化结果", "箱图")
    for dir_path in [bar_chart_dir, line_chart_dir, pie_chart_dir, box_chart_dir]:
        os.makedirs(dir_path, exist_ok=True)  # 如果文件夹不存在，则创建

    # 定义各因素维度和成绩科目列名
    dimensions = ['父母教育水平', '性别', '种族/民族', '午餐', '备考课程']
    score_cols = ['数学成绩', '阅读成绩', '写作成绩']
    bar_width = 0.2  # 每个柱子的宽度

    # 遍历每个因素，绘制柱状图展示其对各科成绩的影响
    for factor in dimensions:
        plt.figure(figsize=(12, 8))  # 设置图表大小
        x_ticks = []
        for row_idx, score_col in enumerate(score_cols):
            grouped_data = data.groupby(factor)[score_col].mean()  # 计算每个因素下各科的平均成绩
            x = [i + row_idx * bar_width for i in range(len(grouped_data))]  # 调整柱子位置
            plt.bar(x, grouped_data.values, width=bar_width, label=score_col)  # 绘制柱状图
            x_ticks.extend(x)
        plt.title(f'{factor}对学生各科成绩的综合影响')  # 设置图表标题
        plt.xlabel(factor)  # 设置x轴标签
        plt.ylabel('成绩')  # 设置y轴标签
        plt.xticks([i + bar_width * (len(score_cols) - 1) / 2 for i in range(len(grouped_data))],
                   grouped_data.index)  # 设置x轴刻度
        plt.legend()  # 显示图例
        safe_factor = factor.replace('/', '_').replace('\\', '_')  # 处理因素名称中的非法字符
        html = mpld3.fig_to_html(plt.gcf())  # 将matplotlib图表转换为HTML
        file_path = os.path.join(bar_chart_dir, f'{safe_factor}对学生各科成绩综合影响柱状图.html')  # 指定保存路径
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)  # 保存HTML文件

    # 绘制各因素对各科成绩影响的综合对比折线图
    for score_col in score_cols:
        plt.figure(figsize=(10, 6))  # 设置图表大小
        for dim in dimensions:
            grouped_data = data.groupby(dim)[score_col].mean()  # 计算每个因素下该科目的平均成绩
            plt.plot(grouped_data.index, grouped_data.values, marker='o', label=dim)  # 绘制折线图
        plt.title(f'各因素对{score_col}的影响综合对比')  # 设置图表标题
        plt.xlabel('影响因素')  # 设置x轴标签
        plt.ylabel(score_col)  # 设置y轴标签
        plt.legend()  # 显示图例
        safe_factor = score_col.replace('/', '_').replace('\\', '_')  # 处理科目名称中的非法字符
        html = mpld3.fig_to_html(plt.gcf())  # 将matplotlib图表转换为HTML
        file_path = os.path.join(line_chart_dir, f"各因素对{safe_factor}影响综合对比折线图.html")  # 指定保存路径
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)  # 保存HTML文件

    # 绘制各科成绩等级分布及及格率动态饼图
    def generate_dynamic_score_pie_chart(score_col, output_dir):
        plt.figure(figsize=(8, 8))  # 设置图表大小
        labels = ['不及格', '及格', '中', '良', '优']  # 定义成绩等级标签
        bins = [0, 60, 70, 80, 90, 100]  # 定义成绩等级区间
        sizes = pd.cut(data[score_col], bins=bins, labels=labels).value_counts().sort_index()  # 计算各等级分布
        pass_rate = (sizes.loc[['及格', '中', '良', '优']].sum() / sizes.sum()) * 100  # 计算及格率
        fig = px.pie(names=labels, values=sizes.values,
                     title=f'{score_col}等级分布及及格率（及格率：{pass_rate:.2f}%）')  # 创建饼图
        fig.update_traces(textposition='inside', textinfo='percent+label')  # 设置饼图文本
        file_path = os.path.join(pie_chart_dir, f"{score_col}等级分布及及格率动态饼图.html")  # 指定保存路径
        fig.write_html(file_path)  # 保存HTML文件

    for score_col in score_cols:
        generate_dynamic_score_pie_chart(score_col, pie_chart_dir)  # 为每个科目生成饼图

    # 绘制多维度多科目成绩箱图
    fig, axes = plt.subplots(len(score_cols), len(dimensions), figsize=(15, 10))  # 创建子图
    for col_idx, dim in enumerate(dimensions):
        for row_idx, score_col in enumerate(score_cols):
            sns.boxplot(x=dim, y=score_col, data=data, ax=axes[row_idx, col_idx])  # 绘制箱图
            axes[row_idx, col_idx].set_title(f'按 {dim} 分类的{score_col}箱型图')  # 设置子图标题
            axes[row_idx, col_idx].set_xticklabels(axes[row_idx, col_idx].get_xticklabels(), rotation=45)  # 设置x轴标签旋转
    html = mpld3.fig_to_html(fig)  # 将matplotlib图表转换为HTML
    file_path = os.path.join(box_chart_dir, "多维度多科目成绩箱型图.html")  # 指定保存路径
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html)  # 保存HTML文件


# 主函数
def main():
    # 定义数据源文件路径
    source_file_path = os.path.join(_src_folder, 'StudentsPerformance.csv')

    # 调用数据预处理函数
    processed_data = preprocess_data(source_file_path, _analytical_results_output_folder)

    if processed_data is not None:
        # 调用数据分析函数
        analyze_data(processed_data, _analytical_results_output_folder)

        # 调用数据可视化函数
        visualize_data(processed_data, _analytical_results_output_folder)
    else:
        print("数据预处理失败，无法继续进行数据分析和可视化。")


# 判断是否为主程序执行，若是，则运行主函数
if __name__ == "__main__":
    main()
