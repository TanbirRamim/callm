"""The smallest useful callm program.

pip install "callm-toolkit[openai]"
export OPENAI_API_KEY=...
python examples/quickstart.py
callm stats
"""

import os

import openai

from callm import callm

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit(
        "No API key found.\n\n"
        "Set OPENAI_API_KEY in this same terminal window, then run the script again:\n"
        '  macOS/Linux:  export OPENAI_API_KEY="sk-..."\n'
        '  PowerShell:   $env:OPENAI_API_KEY="sk-..."\n\n'
        "No key at all? Try examples/offline_demo.py, which needs none."
    )

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
