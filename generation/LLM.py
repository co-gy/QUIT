from openai import OpenAI


class LLM:
    def __init__(self, base_url, api_key, model, log=False):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        assert model in [model.id for model in self.client.models.list()]
        self.model = model
        self.log = log

    def __call__(self, messages, max_tokens=1024, temperature=1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=1,
            max_tokens=max_tokens,
            extra_body={
                "top_k": -1,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            stream=False
        )
        content = response.choices[0].message.content or ""
        if self.log:
            print(content, flush=True)
        return content


llm = LLM(
    base_url="http://localhost:6000/v1",
    api_key="EMPTY",
    model="gemma",
    log=False
)

messages = [
    {
        "role": "user",
        "content": "hello"
    }
]

print(llm(messages))
