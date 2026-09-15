"""The smallest useful callm program.

pip install "callm[openai]"
export OPENAI_API_KEY=...
python examples/quickstart.py
callm stats
"""

import openai

from callm import callm

client = openai.OpenAI(max_retries=0)  # let callm own retries


@callm(cache=True, retry=3)
def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(ask("In one sentence, why do retries need jitter?"))
    print(ask("In one sentence, why do retries need jitter?"))  # cache hit, $0
