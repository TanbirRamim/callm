"""Extract structured data with automatic repair and a fallback provider.

pip install "callm[anthropic,openai,validation]"
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
python examples/structured_output.py
"""

import anthropic
from pydantic import BaseModel, Field

import callm

client = anthropic.Anthropic(max_retries=0)


class Contact(BaseModel):
    name: str
    company: str | None = None
    role: str | None = None
    topics: list[str] = Field(default_factory=list)


@callm.callm(output_schema=list[Contact], retry=2, fallback=["openai/gpt-4o-mini"], max_cost=0.05)
def extract_contacts(notes: str):
    return client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system="Extract every person mentioned as a JSON array of {name, company, role, topics}.",
        messages=[{"role": "user", "content": notes}],
    )


if __name__ == "__main__":
    notes = (
        "Met Priya Raman (VP Data at Northwind) about pipeline costs and eval tooling. "
        "Later a quick call with Tom from Contoso on SSO."
    )
    for contact in extract_contacts(notes):
        print(contact)
    record = callm.last_call()
    print(f"cost ${record.cost_usd:.5f} via {record.provider}/{record.model}")
