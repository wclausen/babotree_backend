from main import get_readwise_highlights, get_openai_summary


def test_get_readwise_articles():
    print()
    articles = get_readwise_highlights()

def test_get_openai_summary():
    print()
    source_id = 5561801
    response = get_openai_summary(source_id)