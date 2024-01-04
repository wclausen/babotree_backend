import json
import uuid

import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, HighlightSource, Flashcard, FlashcardHighlightSource

TYPESCRIPT_FLASHCARD_INTERFACE = """
interface Flashcard {
    question: string;
    answer: string;
}
interface Flashcards {
    flashcards: Flashcard[];
}
"""

TYPESCRIPT_TOPIC_INTERFACE = """
interface Topic {
    name: string;
    description: string;
}
interface Topics {
    topics: Topic[];
}
"""

together_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)

openai_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('OPENAI_API_KEY'),
)

openai_client = together_openai_client
# model = 'gpt-4-1106-preview'
model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

def extract_json(content):
    """
    We try to find a point in the text that is likely to contain a json object, and then extract it
    :param content:
    :return:
    """
    if "```json" in content:
        json_start = content.find("```json\n")
        json_end = content.find("```", json_start + 1)
        json_str = content[json_start + 7:json_end]
        print(json_str)
        return json_str.strip()
    else:
        # look for first { and last }
        json_start = content.find("{")
        json_end = content.rfind("}")
        json_str = content[json_start:json_end + 1]
        json_str = json_str.replace("\n", "")
        json_str = json_str.replace("\[", "[")
        json_str = json_str.replace("\]", "]")
        print(json_str)
        return json_str.strip()


def insert_flashcard(flashcards_json_dict, source_title, source_highlights):
    db = get_direct_db()
    for flashcard in flashcards_json_dict['flashcards']:
        db_model = Flashcard(
            id=uuid.uuid4(),
            topic=source_title,
            question=flashcard['question'],
            answer=flashcard['answer'],
        )
        db.add(db_model)
        # now, connect the flashcard to the highlights that created it
        for highlight in source_highlights:
            connection_model = FlashcardHighlightSource(
                flashcard_id=db_model.id,
                highlight_id=highlight.id,
            )
            db.add(connection_model)
    db.commit()


def generate_flashcards_for_source(highlight_source_id):
    # get all the highlights for this source
    db = get_direct_db()
    highlight_source = db.query(HighlightSource).filter(HighlightSource.id == highlight_source_id).first()
    highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source_id).all()
    db.close()
    # now we use Mistral to generate the flashcards
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" + "\n".join(
                [highlight.text for highlight in highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "What is the topic of the excerpts? Please respond with a JSON object containing the topics formatted according to the following TypeScript interface:\n" + TYPESCRIPT_TOPIC_INTERFACE
        }
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    # print(response.choices[0].message.content)
    topic_json_str = extract_json(response.choices[0].message.content)
    topic_json_dict = json.loads(topic_json_str)
    print(topic_json_dict)
    for i, highlight in enumerate(highlights):
        adjacent_highlights = highlights[i - 2:i] + highlights[i + 1:i + 3]
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" + highlight.text + "\n".join([x.text for x in adjacent_highlights])
            },
            {
                "role": "assistant",
                "content": "Ok, I'm ready."
            },
            {
                "role": "user",
                "content": "What are the topics of the excerpts? Be detailed and concise."
            }
        ]
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=.25,
            presence_penalty=.25,
        )
        print(response.choices[0].message.content)
        # append the response message and ask more questions
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": "Please generate some question/answer flashcards based on the excerpts and topic."
        })
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=.25,
            presence_penalty=.25,
        )
        print(response.choices[0].message.content)
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": "Please be more concise."
        })
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=.5,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=.45,
            presence_penalty=.45,
        )
        print(response.choices[0].message.content)
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": "That's great. Please convert this list of flashcards to a json object obeys this TypeScript interface definition:\n\n" + TYPESCRIPT_FLASHCARD_INTERFACE
        })
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=.25,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=.45,
            presence_penalty=.45,
        )
        print(response.choices[0].message.content)
        try:
            flashcards_json_str = extract_json(response.choices[0].message.content)
            flashcards_json_dict = json.loads(flashcards_json_str)
            insert_flashcard(flashcards_json_dict, highlight_source.readable_title, [highlight] + adjacent_highlights)
        except Exception as e:
            print("error extracting json from response", e)
            continue
        print("------")
        print("------")


def main():
    # the flashcard generation process is fairly involved and involves a pipeline of several steps
    highlight_source_id = '0a16c1a1-33b3-4777-8d2a-59347d1a985a'
    generate_flashcards_for_source(highlight_source_id)

if __name__ == '__main__':
    main()