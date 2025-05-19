import subprocess
import sys
import re

def get_nvdia_version():
    # 使用 subprocess 来执行命令并逐行捕获输出
    result = subprocess.run(
        ['nvidia-smi'],         # 运行 'nvidia-smi' 命令
        stdout=subprocess.PIPE, # 捕获标准输出
        stderr=subprocess.PIPE, # 捕获标准错误
        text=True               # 返回的是文本格式
    )
    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
    if match:
        text = f'cu{"".join(match.group(1).split("."))}'
        return text # 返回捕获到的版本号
    else:
        return "CUDA Version not found"

if __name__ == '__main__':
    if sys.argv[1] == "--cuda":
        print(get_nvdia_version())
