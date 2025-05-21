from smart_pip import smart_install,set_file_env

current_dir,libs = set_file_env()

try:
    from google import genai
except ImportError:
    smart_install("google.genai",libs,add_path=True)
    from google import genai


from httpx._utils import URLPattern
import os
from tools import read_text

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"current_dir:{current_dir}")

class GeminiEngine:
    key = None
    proxy = None
    isOpened = False
    model = 'gemini-1.5-flash'
    def __init__(self):
        if os.path.exists(f"{current_dir}/api.key"):
            GeminiEngine.key = read_text(f"{current_dir}/api.key").strip()
        if os.path.exists(f"{current_dir}/api.proxy"):
            GeminiEngine.proxy = read_text(f"{current_dir}/api.proxy").strip()
        try:
            self.client = genai.Client(api_key=GeminiEngine.key)
            if GeminiEngine.proxy:
                self.setProxy(GeminiEngine.proxy)
            GeminiEngine.isOpened = True
        except Exception as e:
            logger.info("gemini client init fail")
            logger.exception(e)

    def setProxy(self,proxy):
        for client in [
            self.client._api_client._httpx_client,
            self.client._api_client._async_httpx_client
            ]:
            proxy_map = client._get_proxy_map(proxy, False)
            client._mounts ={
                URLPattern(key): None
                if proxy is None
                else client._init_proxy_transport(
                    proxy,
                    #verify=client_args["verify"],
                )
                for key, proxy in proxy_map.items()
            }

    def close(self):
        pass

    def stream(self,text):
        response = self.client.models.generate_content(
            model= GeminiEngine.model,
            contents=text
        )
        logger.info(response.text)
        return response.text

if __name__ == '__main__':
    pass
