import os
from typing import Any, List, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
db_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

T = TypeVar("T", bound=BaseModel)


def invoke(
    system_prompt: str,
    user_message: Union[str, List[dict]],
    response_format: Optional[Type[T]] = None,
    *,
    temperature: float = 0,
    model: str = "gpt-4o-mini",
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Union[str, T]:
    """
    调用 LLM，支持多种自定义选项。

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息，可为 str 或 vision 多模态 content 列表
        response_format: 若提供 Pydantic 模型，则返回结构化解析结果；否则返回 str
        temperature: 采样温度，默认 0
        model: 模型名，默认 gpt-4o-mini
        max_tokens: 最大生成 token 数
        **kwargs: 其他传给 chat.completions.create 的参数

    Returns:
        当 response_format 为 None 时返回 str；否则返回解析后的 Pydantic 实例
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    common = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        common["max_tokens"] = max_tokens

    if response_format is not None:
        completion = client.beta.chat.completions.parse(
            **common,
            response_format=response_format,
        )
        return completion.choices[0].message.parsed

    response = client.chat.completions.create(**common)
    return response.choices[0].message.content

