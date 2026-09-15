"""A complete mini-project: triage support tickets with an LLM.

Everything callm does is visible in one run: structured output, caching, retries, a spend
limit, PII masking and a cost report.

    pip install "callm-toolkit[openai,validation]"
    export OPENAI_API_KEY=sk-...
    python ticket_triage.py

Run it twice: the second run answers from the cache and costs nothing.
"""

from __future__ import annotations

import openai
from pydantic import BaseModel, Field

import callm

client = openai.OpenAI(max_retries=0)  # callm owns retries

TICKETS = [
    "Hi, I was charged twice for May. My email is dana@example.com, card ending 4242.",
    "The export button crashes the app every time I click it on iOS 18.",
    "Can you explain the difference between the Team and Business plans?",
    "URGENT: our production API keys stopped working 20 minutes ago!!",
    "Hi, I was charged twice for May. My email is dana@example.com, card ending 4242.",  # duplicate
]


class Triage(BaseModel):
    category: str = Field(description="billing, bug, question, or outage")
    urgency: int = Field(ge=1, le=5)
    summary: str
    needs_human: bool


@callm.callm(
    name="triage",  # shows up in `callm stats --by function`
    cache=True,  # identical tickets are answered once
    retry=3,  # rate limits and 5xx are retried with backoff
    max_cost=0.02,  # refuse any single call above 2 cents
    block_pii=True,  # emails and card numbers never reach OpenAI
    output_schema=Triage,  # returns a validated Triage object
)
def triage(ticket: str) -> Triage:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You triage customer support tickets. Reply with JSON: "
                    "category (billing|bug|question|outage), urgency 1-5, "
                    "summary (one sentence), needs_human (true/false). "
                    "Personal data may appear as placeholders like [EMAIL_1]; keep them as-is."
                ),
            },
            {"role": "user", "content": ticket},
        ],
    )


def main() -> None:
    for ticket in TICKETS:
        result = triage(ticket)
        record = callm.last_call()
        flags = []
        if record and record.cache_hit:
            flags.append("cached, $0")
        if record and record.pii_redactions:
            flags.append(
                "masked " + ", ".join(f"{k} x{v}" for k, v in record.pii_redactions.items())
            )
        if record and record.retries:
            flags.append(f"retried {record.retries}x")

        print(f"\n{ticket[:70]}{'...' if len(ticket) > 70 else ''}")
        print(f"  → {result.category:8} urgency {result.urgency}  human={result.needs_human}")
        print(f"    {result.summary}")
        if flags:
            print(f"    [{' · '.join(flags)}]")

    print("\nWhat it cost:\n")
    for row in callm.stats(by="model"):
        print(
            f"  {row.key:24} {row.calls} calls  ${row.cost_usd:.5f}  "
            f"cache hits {row.cache_hits}  saved ${row.saved_usd:.5f}"
        )
    print("\nRun `callm stats` or `callm calls` any time to see this again.")


if __name__ == "__main__":
    main()
