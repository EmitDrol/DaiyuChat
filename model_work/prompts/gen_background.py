prompt_template = """以第二人称(将黛玉称呼为你)，即假设你在和林黛玉对话，简单总结告诉林黛玉这幕情形的背景，注意仅仅是背景，不要透露这幕情形里发生的任何事情，也不要说你在介绍背景，实现沉浸式对话。注意请使用符合当时年代的语言。<这幕情形>{}</这幕情形>。以"现在，"开头。并以```text```包裹最终输出。"""

from data_work.data_utils.extractor import text_extractor
prompt_template_extracor = text_extractor