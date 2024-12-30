<h2 id="pip使用说明" style="text-align: center;">`pip`使用说明</h2><br>

本项目依赖一些Python第三方库来实现相应功能，你可以通过 `pip` 命令来安装这些依赖库，以下是具体地使用步骤和相关说明：

### 一、安装 `requirements.txt` 中列出的依赖库
在项目的根目录（包含 `main.py` 文件的目录）下，打开命令行终端（Windows下可以使用 `cmd` 或者 `PowerShell`，Linux 和 macOS 可以使用自带的终端），运行以下命令来一次性安装所有依赖库：

```bash
 pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt
```
这个命令会读取当前目录下的 requirements.txt 文件，然后自动下载并安装文件中列出的各个 Python 库及其对应的版本（如果指定了版本的话）。

### 二、单独安装某个库
如果你只想安装某个特定的库，例如只安装 `pandas` 库，可以使用以下命令：

```bash
pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pandas
```

如果需要指定版本安装，比如安装 `pandas` 的 `1.5.3` 版本，则运行：

```bash
pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pandas~=0.23.4
```

### 三、更新库
有时候，你可能想要更新已安装的库到最新版本，使用以下命令可以实现对单个库的更新，例如更新 `matplotlib`：

```bash
pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --upgrade matplotlib
```

若要更新 `requirements.txt` 文件中列出的所有库到最新版本，可以使用：

```bash
pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --upgrade -r requirements.txt
```

请注意，在更新库时可能会出现兼容性问题，尤其是当项目中的代码依赖于特定版本的库功能时，所以在更新前最好备份项目或者在测试环境中先进行更新测试，确保项目能正常运行。

### 四、查看已安装的库及其版本
你可以通过以下命令查看当前Python环境中已经安装的所有库及其版本信息：

```bash
pip list
```

该命令会列出所有已安装的库以及对应的版本号，方便你确认依赖库是否正确安装以及查看当前环境的状态。

### 五、卸载某个库
如果你需要卸载某个库，例如卸载 `numpy`，可以使用以下命令：
```bash
pip uninstall numpy
```
该命令会提示你确认是否要卸载，输入 `y` 确认后即可卸载。
如果你需要卸载多个库，可以使用以下命令：
```bash
pip uninstall numpy pandas matplotlib
```
该命令会依次提示你确认是否要卸载每个库，输入 `y` 确认后即可依次卸载。<br><br>
以上就是 `pip` 命令的使用说明，希望对您有所帮助。