import os,sys
import re
import subprocess,psutil
import time
from tools import clean,cleanAll
import emoji
work_path = os.getcwd()
import shutil
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
if not os.environ.get("OLLAMA_MODELS"):
    os.environ["OLLAMA_MODELS"] = f"{work_path}/models/ollama_models"
#注意环境变量设置一定要在前面
from ollama import ChatResponse,Client
import threading
import logging
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

translate_format = "将下面的话翻译成{1}，直接输出结果，不要多余的文字：{0}。"

def remove_think_tags(text):
    # 使用正则表达式删除 <think> 标签及其内部内容（非贪婪匹配）
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

class OllamaEngine:
    isOpened = False
    model = None
    """Generate a msg."""
    def __init__(self):
        self.model = OllamaEngine.model if OllamaEngine.model else None

        ollama_bin = shutil.which("ollama")

        if ollama_bin:
            self.ollama_bin = ollama_bin
        else:
            self.ollama_bin = f"{work_path}/libs/ollama/ollama.exe"

        logger.info(f"ollama bin is '{self.ollama_bin}'")
        logger.info(f"ollama OLLAMA_MODELS is '{os.environ['OLLAMA_MODELS']}'")
        
        self.kill_ollama_exes()

        if not self.VerifyServerClose():
            logger.info("kill_ollama_exe fail")
            return

        if self.OpenServer():
            logger.info(f"open ollama serve success")
            self.client = Client(
                  host=os.environ["OLLAMA_HOST"],
                  headers={'x-some-header': 'some-value'}
                )
            try:
                self.models = [model.model for model in self.client.list().models]
                if self.models:
                    logger.info(f"models:{self.models}")
                    if self.model and self.model in self.models:
                        pass
                    else:
                        logger.info(f"{'specified model '+self.model+' no found'if self.model else 'no specified model found'},so use first on list [{self.models[0]}]")
                        self.model = self.models[0]
                else:
                    logging.info(f"no model found in ollama list")
            except Exception as e:
                logger.exception(e)
            if self.model:
                try:
                    text = self.chat(f"repeat the sentence without other words:{self.model} connect success",show_think=False)
                    #logger.info(text)
                    self.isOpened = True
                    return
                except Exception as e:
                    logging.info(f"connect model {self.model} fail,try to close the proxy")
                    logger.exception(e)
        if self.isOpened == False:
            logger.info("ollama init fail")
            #self.close()

    def VerifyServerOpen(self,tik = 1,timeout=10):
        #logger.info("VerifyServerOpen")
        import time
        import requests
        for _ in range(timeout):
            time.sleep(tik)
            try:
                r = requests.get("http://127.0.0.1:11434", timeout=2)
                if r.status_code == 200 and r.text == "Ollama is running":
                    return True
            except:
                continue
        logger.debug("Ollama start timeout")
        return False

    def VerifyServerClose(self, tik=1, timeout=10):
        #logger.info("VerifyServerClose")
        import time
        import requests
        for _ in range(timeout):
            time.sleep(tik)
            try:
                r = requests.get("http://127.0.0.1:11434", timeout=2)
                # 如果仍然返回 200 且内容正确，说明服务还在运行
                if r.status_code == 200 and "Ollama is running" in r.text:
                    continue
            except:
                # 请求失败，说明服务已关闭
                return True
        # 超时仍然未关闭
        return False

    def OpenServer(self,tik = 1,timeout=30):
        # 后台运行 Ollama 服务
        self.process = subprocess.Popen(
            [self.ollama_bin, "serve"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            #text=True,
            shell=False)

        def _log_thread(pipe, level="info"):
            text_pipe = io.TextIOWrapper(pipe, encoding='utf-8', errors='ignore')
            for line in iter(text_pipe.readline, ''):
                line = line.strip()
                getattr(logger, level)(f"[ollama {level}] {line}")

        threading.Thread(target=_log_thread, args=(self.process.stdout, "info")).start()
        threading.Thread(target=_log_thread, args=(self.process.stderr, "debug")).start()
        return self.VerifyServerOpen()

    def chat(self,content,return_think=False,show_think=True):
        response: ChatResponse = self.client.chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': content,
        },
        ])
        text = response['message']['content']
        text = emoji.demojize(text)
        result = text if return_think else remove_think_tags(text)
        show =  text if show_think else remove_think_tags(text)
        logger.info(show)
        return result 

    def stream(self,content,return_think=False):
        stream = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': content}],
            stream=True,
        )
        # 逐块打印响应内容
        parts = []
        #emoji compatible
        for chunk in stream:
            part = chunk['message']['content']
            part = emoji.demojize(part)
            print(part,end="",flush=True)
            parts.append(part)
        print("\n",end="",flush=True)
        text = "".join(parts)
        #logger.info(text)
        result = text if return_think else remove_think_tags(text)
        #logger.info(result)
        return result

    def translate(self,text,lang="en"):
        text = translate_format.format(text,lang)
        return self.stream(text)

    def kill_ollama_exes(self):
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                try:
                    logger.info(f"Killing: {proc.pid} - {proc.name()}")
                    proc.kill()
                except Exception as e:
                    logger.info(f"Error killing process: {e}")

    def close(self):
        self.kill_ollama_exes()
        #self.kill_ollama_exes()
        if not self.VerifyServerClose():
            logger.info(f"kill fail,try again")
            self.kill_ollama_exes()
        else:
            clean(self,attr="process")
            logger.info(f"Successfully terminated Ollama processes")
        
        # 释放显存
if __name__ == '__main__':
    from tools.logger import setup_logging
    setup_logging()
    text = OllamaEngine()
    if text.isOpened:
        text.stream("hello")
    text.close()