<h1 align="center">基于`Python`的学生成绩数据分析与可视化项目</h1>

## 一、项目概述

本项目旨在对学生成绩数据进行全面的分析与可视化展示。通过读取包含学生各项信息（如性别、种族/民族、父母教育水平、午餐情况、备考课程以及数学、阅读、写作成绩等）的CSV数据文件，实现以下主要功能：

1. **数据预处理**：备份原始数据集，处理数据中的缺失值（采用中位数填充数学、阅读、写作成绩列的缺失值），并将列名汉化，方便查看与理解，最后将处理后的数据保存至指定文件夹。
2. **数据分析**：针对不同维度（种族、性别、父母教育水平、午餐类型、测试准备课程）计算平均成绩，并输出相应的分析结果。
3. **数据可视化**：利用多种图表类型（箱型图、饼图、直方图、散点图等）直观展示数据特征，且所有可视化图表均会转换为HTML文档保存，便于查看与分享。

本项目使用了Python中常用的数据处理与可视化相关库，包括 `pandas`、`matplotlib`、`seaborn` 以及 `mpld3` 等。

## 二、项目结构

```bash
Visualization-of-student-performance-data
│
├── .venv - 虚拟环境目录，用于隔离项目的Python依赖
│   ├── Lib - 存放Python库文件
│   ├── Scripts - 存放可执行脚本
│   ├── .gitignore - Git忽略文件，用于指定哪些文件或目录不应该被Git版本控制
│   └── pyvenv.cfg - 虚拟环境的配置文件
│
├── annexes - 附件目录，用于存放项目的额外文档
│   ├── PIP使用说明.md - 关于如何使用PIP的说明文档
│   ├── 代码分析.md - 对项目代码的分析文档
│   ├── 提取的问题.md - 从代码中提取的问题列表
│   └── 提取的问题_参考答案.md - 对提取问题的答案或参考解答
│
├── src - 源代码目录，包含项目的主要代码和数据
│   ├── AnalyticalResults - 存放分析结果的目录
│   ├── data_backup - 用于备份数据的目录
│   ├── data_source - 存放数据源的目录
│   ├── main.py - 项目的主程序文件
│   └── requirements.txt - 列出项目依赖的Python库及其版本的文件
│
├── LICENSE - 项目的许可证文件，说明项目的使用条款
├── README.md - 项目的自述文件，通常包含项目简介、安装和使用说明等信息
```

## 三、技术栈

**本项目使用以下技术栈：**

- **`Python`**: 编程语言，用于数据处理和分析。
- **`Pandas`**: 用于数据操作和分析。
- **`Matplotlib & Seaborn`**: 用于数据可视化。
- **`MPld3`**: 用于将Matplotlib图表转换为HTML格式，便于在网页上展示。
- **`Git`**: 用于版本控制。

## 四、使用方式

1. **克隆项目**:
   ```bash
   git clone https://github.com/dcyyd/Visualization-of-student-performance-data.git
   ```

2. **创建并激活虚拟环境**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

4. **运行主程序**:
   ```bash
   python src/main.py
   ```

5. **查看结果**:
   处理后的数据和生成的可视化图表将保存在`src/output`目录下。

## 五、作者

ChangYou Dou (dcyyd_kcug@yeah.net)

## 六、许可证

本项目遵循MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 七、联系和支持

如果您有任何问题或需要帮助，请通过以下方式联系我：

- **Email**: <a href="mailto:dcyyd_kcug@yeah.net">dcyyd_kcug@yeah.net</a>
- **Phone**：<a href="tel:+17633963626">17633963626</a>
- **Blog**: [https://www.kfufys.top](https://www.kfufys.top)
- **GitHub**: [https://github.com/dcyyd](https://github.com/dcyyd)