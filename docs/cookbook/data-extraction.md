# Data extraction

Turn unstructured invoices into validated records in bulk, with automatic repair of malformed
output, a hard spend cap for the batch and a cost report at the end.

```python
from concurrent.futures import ThreadPoolExecutor
import contextvars
from datetime import date
from decimal import Decimal

from google import genai
from pydantic import BaseModel, Field

import callm

client = genai.Client()


class LineItem(BaseModel):
    description: str
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    issued: date
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    items: list[LineItem]
    total: Decimal


@callm.callm(
    name="extract.invoice",
    output_schema=Invoice,
    validation_retries=2,
    cache=True,
    retry=4,
    block_pii=True,
)
def extract_invoice(text: str):
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Extract this invoice as JSON matching the Invoice schema.\n\n{text}",
        config={"response_mime_type": "application/json", "max_output_tokens": 2048},
    )


def extract_all(documents: list[str], limit_usd: float = 5.00) -> list[Invoice | None]:
    results: list[Invoice | None] = []
    with callm.budget(limit_usd, name="invoice-batch") as batch:
        with ThreadPoolExecutor(max_workers=8) as pool:
            # Copy the context so every worker thread sees the batch budget.
            futures = [
                pool.submit(contextvars.copy_context().run, extract_invoice, doc) for doc in documents
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except callm.OutputValidationError as exc:
                    print("could not extract:", exc.errors[:3])
                    results.append(None)
                except callm.BudgetExceeded:
                    results.append(None)
    print(f"batch spent ${batch.spent:.4f}")
    return results
```

After the run:

```console
$ callm stats --function extract.invoice --since 1h
$ callm calls --function extract.invoice --limit 5
```

**Notes**

- `response_mime_type="application/json"` asks Gemini for JSON; callm still validates types,
  patterns and constraints and re-asks with precise errors when something is off.
- Email addresses and phone numbers on invoices are masked before sending. Remove `block_pii`
  if you need to extract them.
- Identical documents are served from the cache, which makes re-running a partially failed batch
  cheap.
