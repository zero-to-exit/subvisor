import os
import sys
from openai import OpenAI


def main() -> None:
    
    client = OpenAI(
    api_key="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"
    )

    response = client.responses.create(
    model="gpt-5-nano",
    input="write a haiku about ai",
    store=True,
    )

    print(response.output_text);


if __name__ == "__main__":
    main()


