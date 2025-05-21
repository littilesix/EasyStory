@echo off
:: 设置命令提示符样式（Prompt format）
prompt (installer) $P$G

:: 获取当前工作路径（Set work path to current directory）
set workPath=%~dp0
set pythonPath=%workPath%\env
set python=%pythonPath%\python.exe
set pip=%pythonPath%\scripts\pip.exe

:: 创建缓存目录（Create installation/cache folders）
mkdir installer 2>nul
mkdir installer\cache 2>nul

:: 如果 Python 已存在则跳过安装（Check if Python already exists）
if exist "%python%" (
    echo Python already exists at %python%
) else (
    echo Python not found, installing...

    :: 下载 Python 3.10 嵌入版（Download embedded Python）
    curl -L -o installer\cache\python3.10.6.zip https://mirrors.aliyun.com/python-release/windows/python-3.10.6-embed-amd64.zip

    mkdir %pythonPath% 2>nul
    tar -xf installer\cache\python3.10.6.zip -C %pythonPath%
)

:: 显示 Python 版本（Show Python version）
echo PYTHON VERSION:
%python% --version

:: 如果 pip 存在则跳过安装（Check for pip）
if exist "%pip%" (
    echo pip already exists at %pip%
) else (
    :: 安装 pip（Install pip）
    %python%  installer/get-pip.py

    :: 修改 _pth 文件，使嵌入式 Python 能使用 site-packages（Enable site-packages）
    echo import site>>env/python310._pth
    echo ../modules>>env/python310._pth
    echo ../modules/lora/sd-scripts>>env/python310._pth
    echo ../installer>>env/python310._pth
    echo ../scripts>>env/python310._pth
)

:: 设置环境变量 PATH（Temporarily add Python to path）
set path=%pythonPath%;%pythonPath%/scripts;%PATH%

:: 设置镜像（Set Aliyun mirrors）
set MirrorHost=mirrors.aliyun.com
set MirrorSimple=https://%MirrorHost%/pypi/simple
set aliTorchMirror=https://mirrors.aliyun.com/pytorch-wheels

echo upgrade the pip ......
python -m pip install -i %MirrorSimple% --upgrade pip --trusted-host %MirrorHost%

:: pip 安装失败处理（Handle pip failure）
if %errorlevel% neq 0 (
    echo Pip installation failed. You can run this bat file again...
    pause
)

:: 获取 CUDA 版本（Detect CUDA version）
for /f "tokens=*" %%i in ('python installer\get-info.py --cuda') do set cuda=%%i

:: 若未检测到 CUDA 则报错退出（No CUDA found = abort）
if [%cuda%]==[] (
    echo.
    echo ===========================
    echo [ERROR] No CUDA device found!
    echo This program needs a CUDA-capable GPU.
    echo ===========================
    echo.
    pause
    exit /b
)

:: 输出 CUDA 版本（Show CUDA version）
echo.
echo ===========================
echo CUDA VERSION:
echo %cuda%
echo ===========================
echo.

prompt (easyStory-%cuda%) $P$G
:: 安装 PyTorch 相关依赖（Install torch with correct CUDA version）
echo install torch environments
pip install torch torchvision torchaudio -f %aliTorchMirror%/%cuda% --trusted-host %MirrorHost%
if %errorlevel% neq 0 (
    echo Torch installation failed becasue connect timeout. You can run this bat file again...
    pause
    exit /b
)

echo.
echo ===========================
echo [SUCCESS] all pytorch environments is ready
echo ===========================
echo.

:: ========================
:: 嵌入式 Python 不能通过 pip 安装 filterpy，因此手动下载解压
:: Fix filterpy compatibility with embedded Python
:: ========================
rmdir /s /q "modules\filterpy"

curl -L -o installer\cache\filterpy.zip https://mirrors.aliyun.com/pypi/packages/f6/1d/ac8914360460fafa1990890259b7fa5ef7ba4cd59014e782e4ab3ab144d8/filterpy-1.4.5.zip

tar -xf "installer\cache\filterpy.zip" -C "installer\cache"

if %errorlevel% neq 0 (
    echo down filterpy.zip failed becasue connect timeout. You can run this bat file again...
    pause
    exit /b
)

move "installer\cache\filterpy-1.4.5\filterpy" "modules\filterpy"

:: 安装其他依赖（Install Python requirements）
echo install requirements
pip install -i %MirrorSimple% -r installer/requirements.txt --trusted-host %MirrorHost%
echo end requirements install

if %errorlevel% neq 0 (
    echo install requirements failed. You can run this bat file again...
    pause
    exit /b
)

:: 安装 sd-scripts 为本地包（Install sd-scripts as editable module）
cd modules/lora/sd-scripts
pip install -e .
cd ../../..

:: 清理 filterpy（Cleanup filterpy workaround）
rmdir /s /q "modules/filterpy"


echo.
echo ===========================
echo [SUCCESS] all easyStory packages is ready
echo ===========================
echo.
:: 下载模型（Download pre-trained models）
python installer/get_models.py
if %errorlevel% neq 0 (
    echo something exception happened
    pause
    exit /b
)
echo.
echo ===========================
echo [SUCCESS] all easyStory models is ready
echo ===========================
echo.
:: 清理缓存（Final cleanup）
rmdir /s /q "installer/cache"
echo.
echo ===========================
echo [SUCCESS] finish all install
echo ===========================
echo.
cmd
