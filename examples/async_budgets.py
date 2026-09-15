"""Concurrent async calls under a shared session budget, with PII masking.

pip install "callm-toolkit[openai]"
export OPENAI_API_KEY=...
python examples/async_budgets.py
"""

import asyncio

import openai

import callm

client = openai.AsyncOpenAI(max_retries=0)


@callm.callm(retry=3, block_pii=True, detect_injection=True)
async def classify(ticket: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        messages=[
            {
                "role": "system",
                "content": "Classify the ticket as billing, bug, or other. One word.",
            },
            {"role": "user", "content": ticket},
        ],
    )
    return (response.choices[0].message.content or "").strip().lower()


async def main() -> None:
    tickets = [
        "I was charged twice, my card ends 4242 4242 4242 4242",
        "The export button crashes the app on iOS 18",
        "Can you email me at sam@example.org about the roadmap?",
    ]
    async with callm.budget(0.01) as session:
        labels = await asyncio.gather(*(classify(t) for t in tickets))
    for ticket, label in zip(tickets, labels, strict=True):
        print(f"{label:>8}  {ticket}")
    print(f"session spent ${session.spent:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
