import openai
from openai import OpenAI
from typing import Dict, Any

class BaseOpenaiModel:
    """
    基础类，用于封装 OpenAI API 的调用逻辑。
    """

    def __init__(self, base_url: str,api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        初始化 OpenAI API 客户端。

        :param api_key: OpenAI 的 API 密钥。
        :param model_name: 使用的模型名称，默认为 "gpt-3.5-turbo"。
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """
        调用 OpenAI API 完成文本生成任务。

        :param prompt: 输入的提示文本。
        :param max_tokens: 生成的最大 token 数量，默认为 4096。
        :param temperature: 控制生成文本的随机性，范围为 0.0 到 1.0，默认为 0.7。
        :param kwargs: 其他可选参数，例如 `stop`, `n` 等。
        :return: 包含 API 响应的字典。
        """
        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return self._extract_response_text(response=response)
        except Exception as e:
            raise RuntimeError(f"调用 OpenAI API 失败: {e}")

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """
        从 OpenAI API 响应中提取生成的文本。

        :param response: OpenAI API 返回的响应字典。
        :return: 提取的生成文本。
        """
        if len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("无法从响应中提取生成的文本，请检查 API 响应格式。")

# 示例使用
if __name__ == "__main__":
    
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_KEY = "sk-60c35b95f17f42a0bd50452648b7e07e"

    # 初始化模型
    model = BaseOpenaiModel(base_url=BASE_URL,api_key=API_KEY, model_name="qwen2-72b-instruct")

    # 调用 API
    prompt = """任务描述：
你是一名剧本处理助手，负责将单幕剧本的内容转换为 JSON 对象列表格式。

输入信息：
用户会提供一段单幕剧本内容。剧本中可能包含对话、旁白或舞台指示。

输出要求：
请将剧本内容转换为一个 JSON 对象列表，每个对象包含以下两个字段：
1. role：表示发言者的角色，可能是角色名（如 'Alice'）、'Narrator'（旁白）。
2. content：表示该角色或旁白的具体内容。
当完成JSON 对象列表输出后进行一次反思，检查是否满足约束条件并再次输出反思后的JSON 对象列表。

JSON 列表的格式如下：
```json
[
  {"role": "<角色名>", "content": "<内容>"},
  {"role": "<角色名>", "content": "<内容>"}
]
```
确保每个发言者或旁白都对应一个独立的对象，并按照剧本中的顺序排列。

约束条件：
- 如果剧本中有明确的角色名，请将其作为 `role` 的值。
- 如果是旁白（例如描述场景或背景），请将 `role` 设置为 'Narrator','Narrator'message 里不能有角色的动作或心理活动，动作或心理活动需要嵌入到角色的content中。
- 如果某条message没有话语，请根据上下文，为该条message添加适量的合理的话语。
- 角色(不包括Narrator)的动作需要嵌入到角色(不包括Narrator)的content中，并用<action></action>标签包裹，若单条对话缺少动作描写，你可以适当自由发挥，嵌入适量且合理的动作。
- 角色(不包括Narrator)的心理活动需要嵌入到角色(不包括Narrator)的content中，并用<thinking></thinking>标签包裹，若单条对话没有角色内心所想，你可以适当自由发挥，嵌入适量且合理的心理活动。
- 角色的话语不要用任何标签包裹，也不要用引号包裹。
- 不能出现相邻message的role(包括Narrator)相同的情况。

示例

单幕剧本：
正是暮春时节，草树争茂，各种颜色的花开得正盛，也正开始凋落…… \n　　一树桃花开得喷火蒸霞一般。树下一方长石上坐着宝玉，正聚精会神地读着一本书…… \n　　字幕（叠）: \n　　第七集 叔嫂染恙\n　　\n　　一阵清风吹来，树上桃花纷纷洒落，落了宝玉满头、满身、满书，四周地上也落满鲜美的花瓣…… \n　　宝玉轻轻站起身来，双手兜着衣襟，想往地上抖落，忽见地上花瓣落在杂草和泥土中。\n　　宝玉皱一下眉，摇摇头。\n　　耳边传来潺潺流水声。\n　　宝玉转头，清彻的沁芳溪水泛着浪花活泼泼地向下流去……\n　　宝玉露出欣喜的微笑，轻轻走烈溪水边，小心地将花瓣抖落在溪水中。\n　　花瓣随水漂转流去……\n　　宝玉欣慰地笑了。\n　　你在这里作什么？\n　　宝玉一回头，原来是黛玉笑盈盈地站在他身后，肩上担着花锄，锄上挂着花囊，手内拿着花帚。\n　　宝玉高兴地：好，好，快把这些花扫起来，撩在水里……\n　　黛玉：撩在水里不好，流到那脏的臭的地方又把花糟踏了，不如装进这绢袋里，埋起来，岂不更干净。\n　　宝玉：好，好。我放下书帮你收拾。\n　　黛玉看着宝玉手中的书：什么书？\n　　宝玉慌忙把书藏在身后，笑着：不过是《中庸》、《大学》之类。\n　　黛玉笑着横他一眼：你又在我跟前弄鬼。伸出手来：趁早给我瞧瞧，别让我费事！ \n　　宝玉笑着：好妹妹，若论你我是不怕的。你看了，千万别告诉别人。真真是好文章，你要看了，连饭都不想吃呢。一面说，一面把书递过去。\n　　黛玉把花具放下，接过书来，见书皮上是《西厢记》三个字。\n　　黛玉一笑，坐在山石上翻看。\n　　宝玉也凑过来，并排坐着，不时指着书中某句提示黛玉。\n　　桃花瓣象红雨般向宝玉和黛玉身上洒落……

正确输出(反思前错误输出省略)：

<反思前错误输出>

反思与调整
检查相邻 role 是否重复 ：
检查发现相邻的 role 相同的情况，将相邻的message整合。
动作和心理活动标签是否正确 ：
所有动作都已用 <action></action> 包裹。
所有心理活动都已用 <thinking></thinking> 包裹。
旁白内容是否合理 ：
旁白仅包含描述性内容，未混入角色的动作或心理活动。
自由发挥是否适度 ：
在适当位置添加了合理的心理活动和动作描写，例如宝玉看到花瓣时的心理活动、黛玉接书时的好奇等。
是否某条message不包含话语 ：
检测到若干条message不包含话语，将适量合理添加。
话语是否被引号包裹 :
检测到若干话语被引号包裹，将更正。

最终输出：

```json
[
  {"role": "Narrator", "content": "正是暮春时节，草树争茂，各种颜色的花开得正盛，也正开始凋落……一树桃花开得喷火蒸霞一般。这景象充满了生机与衰败交织的美感，仿佛预示着某种微妙的情感变化。树下一方长石上坐着宝玉，正聚精会神地读着一本书……一阵清风吹来，树上桃花纷纷洒落，落了宝玉满头、满身、满书，四周地上也落满鲜美的花瓣……"},
  {"role": "宝玉", "content": "<action>轻轻站起身来，双手兜着衣襟，想往地上抖落,忽见地上花瓣落在杂草和泥土中。</action>哎……<thinking>这些花瓣不该被随意丢弃。</thinking>"},
  {"role": "Narrator", "content": "耳边传来潺潺流水声。沁芳溪水的清澈似乎给了宝玉灵感，他心中暗想，若将这些花瓣放入水中，或许能让它们更干净地离去。水流的声音宛如低语，仿佛在回应他的想法。"},
  {"role": "宝玉", "content": "<action>转头看向清彻的沁芳溪水，露出欣喜的微笑，轻轻走到溪水边，小心地将花瓣抖落在溪水中。</action>这般甚好！<thinking>这样可以让花瓣更干净地离去。</thinking>"},
  {"role": "Narrator", "content": "花瓣随水漂转流去……宝玉欣慰地笑了。他感到自己做了一件有意义的事，这些花瓣仿佛承载了他的情感，随着溪水远去。他望着溪水，眼中透露出一丝满足。"},
  {"role": "黛玉", "content": "<action>笑盈盈地站在宝玉身后，肩上担着花锄，锄上挂着花囊，手内拿着花帚。</action>你在这里作什么？"},
  {"role": "宝玉", "content": "<action>高兴地回答</action>好，好，快把这些花扫起来，撩在水里……"},
  {"role": "黛玉", "content": "<action>摇头表示反对</action>撩在水里不好，流到那脏的臭的地方又把花糟踏了，不如装进这绢袋里，埋起来，岂不更干净。"},
  {"role": "宝玉", "content": "<action>点头赞同</action>好，好。我放下书帮你收拾。"},
  {"role": "黛玉", "content": "<action>看着宝玉手中的书，好奇地问</action>什么书？"},
  {"role": "宝玉", "content": "<action>慌忙把书藏在身后，笑着</action>不过是《中庸》、《大学》之类。<thinking>这书可不能让妹妹看见。</thinking>"},
  {"role": "黛玉", "content": "<action>笑着横他一眼，伸出手来</action>趁早给我瞧瞧，别让我费事！"},
  {"role": "宝玉", "content": "<action>笑着递过书</action>好妹妹，若论你我是不怕的。你看了，千万别告诉别人。真真是好文章，你要看了，连饭都不想吃呢。"},
  {"role": "Narrator", "content": "黛玉把花具放下，接过书来，见书皮上是《西厢记》三个字。她心中暗自觉得有趣，这本书的内容向来被视作闺中禁书，没想到宝玉竟然偷偷阅读。她的嘴角微微扬起，带着几分戏谑。"},
  {"role": "黛玉", "content": "<action>一笑，坐在山石上翻看</action>嗯……"},
  {"role": "Narrator", "content": "桃花瓣像红雨般向宝玉和黛玉身上洒落……这一刻，仿佛时间静止，周围的一切都变得模糊，只剩下他们两个人。微风拂过，带来淡淡的花香。这一刻，他们沉浸在彼此的世界中。"}
]
```

用户提供的单幕剧本：
黛玉看书入了神。\n　　宝玉笑问：妹妹，你说好不好？\n　　黛玉笑着点头。\n　　宝玉顽皮地指着自己鼻子：我就是个‘多愁多病身’……\n　　黛玉抬头向宝玉一皱眉。\n　　宝玉接着：你就是那‘倾国倾城貌’。\n　　黛玉霍地站起身来，微腮带怒，薄面含嗔，指着宝玉：你这该死的胡说！你弄这些淫词艳曲来看，还学了这些混话来欺负我，我告诉舅舅和舅母去。说着眼圈一红，转身就走。\n　　宝玉慌忙上前拉住：好妹妹，千万饶我这一遭。我要有心欺负你，指着旁边的水池：明儿我掉在池子里，叫癞头鼋吞了，变个大王八。等你做了一品夫人，病老归西的时候，我往你坟上驼一辈子碑去。\n　　黛玉嗤的一声笑了：瞧你吓得这个样，还胡说。又揉揉眼睛，笑着：呸，原来是苗而不秀，是个银样镴枪头！\n　　宝玉指着黛玉笑：你说我，你这话呢？我也告诉去。\n　　黛玉笑：你会过目成诵，我就不能一目十行？忽听远处隐隐传来锣鼓声和琴笛声：这是哪儿？ \n　　宝玉高高兴兴地收起书：这是梨香院小戏子们在练戏。咱们快把花埋了吧。说着拿起花帚要扫花……\n　　到处找不着你，原来在这儿。袭人急匆匆走来：快回去换衣服。老爷传进话说：北静王爷找你去会友，快走！\n　　宝玉抱歉似地看了一眼黛玉，随袭人走了。\n　　黛玉没情没趣地看着他们走去，也扛起花锄，慢慢往回走。……\n　　黛玉刚走了几步，来到沁芳闸边，忽然从梨香院方向隐隐传来小戏子练唱昆曲的声音：\n　　却原来婉紫嫣红开遍，\n　　似这般都付与断井颓垣……\n　　黛玉住步，侧耳细听。\n　　缠绵悱恻的唱曲声继续传来：\n　　良辰美景奈何天，\n　　赏心乐事谁家院？\n　　……\n　　落红成阵，流水悠悠……\n　　黛玉不觉心动神摇，如醉如痴，站立不住……\n　　如怨如慕的唱曲声继续传来：\n　　则为你如花美眷，\n　　似水流年……\n　　黛玉一蹲身坐在一块山石上，痴痴地俯视着沁芳闸下的流水…… \n　　如泣如诉的唱曲声重复：\n　　则为你如花美眷，\n　　似水流年…… \n　　桃花乱落，水面皆红，旋转飘荡，流下沁芳闸去……\n　　黛玉心痛神痴，两眼噙满泪水。\n　　飘满落花的水不断地流着…… \n　　黛玉的泪水蜿蜒而下……

(PS:请严格按照要求完成任务，否则你会受到惩罚!)


"""
    response = model.call(prompt, max_tokens=2048, temperature=0.7)
    print(prompt ,"\n", response)