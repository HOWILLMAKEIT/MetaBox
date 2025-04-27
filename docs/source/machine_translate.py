import logging
import re
import sys
import time

import httpx
import polib

logging.basicConfig(level=logging.INFO)

pattern = re.compile(r"^(Bases: )?\{py:(obj|mod)\}`.*?`$")  # match {py:obj}`...` or Bases: {py:obj}`...`

template = """Please translate the following text from a Python evolutionary algorithm library's documentation into Chinese.
- Use simple and clear language.
- For specific terms such as class names and function names (e.g. Algorithms, Problems, jit, API), retain their original English form.
- For python code only segments, please do not translate them, return them as they are.
- For references to academic papers, please do not translate them, return them as they are.
- Maintain the same structured format (e.g. `...`, **...**, (...)[...] block) as the original text.
- Maintain the original links and cross-references.
- Only translate the given text, do not expand or add new content.
- The translate for the following words are provided:
  - algorithm: 算法
  - problem: 问题
  - workflow: 工作流
  - population: 种群
  - evolution: 演化
  - fitness: 适应度

**Only return the translated text; no explanation is needed.**
The text to be translated is: {}"""


class TGIBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str,
    ):
        super().__init__()
        url = "https://" + base_url + "/v1/chat/completions"
        self.url = url
        self.api_key = api_key
        self.num_retry = 10
        self.usage_history = []

    def _one_restful_request(self, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]
        data = {
            "messages": messages,
            "stream": False,
            "model": "gpt-4o",
            "max_tokens": 10000,
            "top_p": 0.8,
        }
        for retry in range(self.num_retry):
            try:
                response = httpx.post(self.url, headers=headers, json=data, timeout=30.0)
                json_response = response.json()
            except Exception:
                import traceback

                logging.error(f"Failed to query TGI. Sleep. Retry {retry + 1}...")
                logging.error(traceback.format_exc())
                # network error, sleep and retry
                time.sleep(30)
                continue

            try:
                content = json_response["choices"][0]["message"]["content"]
                usage = json_response["usage"]
                return content, usage
            except Exception:
                logging.error(f"Failed to parse response: {json_response}")
                logging.error(f"{response.text}")
                logging.error(f"{response.json()}")

        logging.error(f"Failed to query TGI for {self.num_retry} times. Abort!")
        raise Exception("Failed to query TGI")

    def query(self, query):
        response = self._one_restful_request(query)

        content, usage = response
        logging.info(f"Received content: {content}")

        return content


class ZhipuAIBackend:
    def __init__(
        self,
        api_key: str,
    ):
        super().__init__()
        self.url = "https://open.bigmodel.cn/api/paas/v3/model-api/glm-z1-flash/invoke"  # 智普清言API端点
        self.api_key = api_key
        self.num_retry = 10
        self.usage_history = []

    def _one_restful_request(self, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,  # 智普清言直接使用API Key
        }
        
        data = {
            "prompt": [{
                "role": "user",
                "content": query,
            }],
            "temperature": 0.8,
            "top_p": 0.8,
            "max_tokens": 10000,
        }
        
        for retry in range(self.num_retry):
            try:
                response = httpx.post(self.url, headers=headers, json=data, timeout=30.0)
                json_response = response.json()
            except Exception:
                import traceback
                logging.error(f"Failed to query ZhipuAI. Sleep. Retry {retry + 1}...")
                logging.error(traceback.format_exc())
                time.sleep(30)
                continue

            try:
                # 根据智普清言API的返回格式调整
                content = json_response["data"]["choices"][0]["content"]
                usage = json_response.get("data", {}).get("usage", {})
                return content, usage
            except Exception:
                logging.error(f"Failed to parse response: {json_response}")
                logging.error(f"{response.text}")

        logging.error(f"Failed to query ZhipuAI for {self.num_retry} times. Abort!")
        raise Exception("Failed to query ZhipuAI")

    def query(self, query):
        response = self._one_restful_request(query)
        content, usage = response
        logging.info(f"Received content: {content}")
        return content


if __name__ == "__main__":
    api_key = sys.argv[1]  # 只需要一个参数：API密钥
    zhipu = ZhipuAIBackend(api_key=api_key)
    import os
    # po_path = os.path.join(os.path.dirname(__file__), "..", "locale/zh/LC_MESSAGES/docs.po")
    po = polib.pofile("/home/hohq/Code/readdocs/MetaBox/docs/source/locale/zh_CN/LC_MESSAGES/docs.po")
    try:
        for entry in po:
            if entry.msgstr and not entry.fuzzy:
                continue

            if pattern.match(entry.msgid) or entry.msgid.startswith("<svg"):
                logging.info(f"Skipping: {entry.msgid}")
                entry.msgstr = entry.msgid
                continue

            query = template.format(entry.msgid)
            logging.info(f"Query: {entry.msgid}")
            tranlated = zhipu.query(query)
            logging.info("\n")
            entry.msgstr = tranlated
            if entry.fuzzy:
                entry.flags.remove("fuzzy")
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        po.save("docs/source/locale/zh/LC_MESSAGES/docs.po")
