"""
This scratch file tries to figure out the average length of a highlight in terms of tokens
that would be parsed by an LLM.

This is helpufl in understanding how many highlights we can reasonably bucket together when sending data
to an LLM to make cards
"""
import tiktoken
from sqlalchemy import func

from app.database import get_direct_db
from app.models import Highlight


def main():
    db = get_direct_db()
    random_highlights = db.query(Highlight).order_by(func.random()).limit(1000).all()
    total_tokens = 0
    tiktoken_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for highlight in random_highlights:
        text_tokens = tiktoken_encoding.encode(highlight.text)
        total_tokens += len(text_tokens)
    print(f"Average number of tokens per highlight: {total_tokens / len(random_highlights)}")

if __name__ == '__main__':
    main()