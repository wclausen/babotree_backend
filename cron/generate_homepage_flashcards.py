import json
import uuid
from typing import List

import numpy as np
import openai
import sklearn.preprocessing
import sqlalchemy
from tenacity import retry, wait_random_exponential, stop_after_attempt

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, HighlightSource, Flashcard, FlashcardHighlightSource, FlashcardGenerationType

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


def insert_flashcard(flashcards_json_dict, source_title, flashcard_generation_type):
    db = get_direct_db()
    for flashcard in flashcards_json_dict['flashcards']:
        db_model = Flashcard(
            id=flashcard['id'] if 'id' in flashcard else uuid.uuid4(),
            topic=source_title,
            question=flashcard['question'],
            answer=flashcard['answer'],
            generation_type=flashcard_generation_type,
        )
        db.add(db_model)
    db.commit()


def insert_flashcard_highlight_connections(flashcards_json_dict, highlights_used_in_flashcards):
    # now, connect the flashcard to the highlights that created it
    db = get_direct_db()
    for flashcard in flashcards_json_dict['flashcards']:
        for highlight in highlights_used_in_flashcards:
            connection_model = FlashcardHighlightSource(
                flashcard_id=flashcard['id'],
                highlight_id=highlight.id,
            )
            db.add(connection_model)
    db.commit()
    db.close()


def generate_basic_front_back_flashcards_for_source(highlight_source_id):
    # get all the highlights for this source
    db = get_direct_db()
    highlight_source = db.query(HighlightSource).filter(HighlightSource.id == highlight_source_id).first()
    highlights = (db.query(Highlight)
                  .filter(Highlight.source_id == highlight_source_id).all())
    highlight_ids_needing_generation = set([x.id for x in get_highlights_not_used_in_flashcard_type(highlight_source_id,
                                                                                                    FlashcardGenerationType.BASIC_FRONT_BACK_FLASHCARD)])
    db.close()
    for i, highlight in enumerate(highlights):
        if highlight.id not in highlight_ids_needing_generation:
            continue
        adjacent_highlights = highlights[i - 2:i] + highlights[i + 1:i + 3]
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" + highlight.text + "\n".join(
                    [x.text for x in adjacent_highlights])
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
            # add uuid to each flashcard, used to connect the flashcard to the highlights that created it
            for flashcard in flashcards_json_dict['flashcards']:
                flashcard['id'] = uuid.uuid4()
            insert_flashcard(flashcards_json_dict, highlight_source.readable_title,
                             FlashcardGenerationType.BASIC_FRONT_BACK_FLASHCARD.value)
            insert_flashcard_highlight_connections(flashcards_json_dict, [highlight] + adjacent_highlights)
        except Exception as e:
            print("error extracting json from response", e)
            continue
        print("------")
        print("------")


def get_source_topics(highlight_source, highlights):
    """
    Asks the Mistral model to identify topics and return a json response according to a typescript interface
    :param highlight_source:
    :param highlights:
    :return:
    """
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
            "content": "What are the main topics of the excerpts? Please identify at most 5 main topics. Each topic should be less than 5 words. Please respond with a JSON object containing the topics formatted according to the following TypeScript interface:\n" + TYPESCRIPT_TOPIC_INTERFACE
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
    try:
        topic_json_str = extract_json(response.choices[0].message.content)
        topic_json_dict = json.loads(topic_json_str)
        print(topic_json_dict)
        return topic_json_dict
    except Exception as e:
        print("error extracting json from response", e)
        return None


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(texts: List[str], model="text-embedding-ada-002") -> List[float]:
    """
    Returns openai embeddings for each
    :param highlights:
    :return:
    """
    embeddings = []
    for text in texts:
        embeddings.append(openai_openai_client.embeddings.create(input=text, model=model).data[0].embedding)
    return embeddings


def get_relevant_term_definition_flashcards(flashcards_json_dict, highlight_source, highlights):
    """
    Returns a list of flashcards that are sufficiently relevant to the main topics of the highlights.
    :param flashcards_json_dict:
    :param highlight_source:
    :param highlights:
    :return:
    """
    topics_json_dict = get_source_topics(highlight_source, highlights)
    if not topics_json_dict:
        print("Could not get topics for this source. Returning all flashcards.")
        return flashcards_json_dict
    # create a list of embeddings/flashcard for each flashcard
    flashcard_embeddings = get_embeddings(
        [flashcard['question'] + "\n" + flashcard['answer'] for flashcard in flashcards_json_dict['flashcards']])
    # normalize the embeddings
    flashcard_embeddings = sklearn.preprocessing.normalize(flashcard_embeddings)
    topics_embeddings = get_embeddings(
        [topic['name'] + "\n" + topic['description'] for topic in topics_json_dict['topics']])
    # normalize the embeddings
    topics_embeddings = sklearn.preprocessing.normalize(topics_embeddings)
    # now we need to compute the cosine similarity between each highlight embedding and each topic embedding
    # and then we need to filter out any flashcards that are sufficiently far away from the overall topics of the highlights
    good_flashcards_json_dict = {
        "flashcards": []
    }
    for flashcard, flashcard_embedding in list(zip(flashcards_json_dict['flashcards'], flashcard_embeddings)):
        # now we need to compute the cosine similarity between the flashcard embedding and each topic embedding
        max_cosine_similarity = -1
        for topic_embedding in topics_embeddings:
            max_cosine_similarity = max(max_cosine_similarity, np.dot(flashcard_embedding, topic_embedding))
        # now we need to compute the average cosine similarity
        if max_cosine_similarity > .82:
            good_flashcards_json_dict['flashcards'].append(flashcard)
    return good_flashcards_json_dict


def get_term_definition_flashcards_from_highlights(highlight_source, highlights):
    TYPESCRIPT_KEY_TERM_INTERFACE = """
    enum TermType {
        GENERAL = 'GENERAL',
        API_COMMAND = 'API_COMMAND',
        API_CLASS = 'API_CLASS',
        API_METHOD = 'API_METHOD',
    }
    interface KeyTerm {
        term: string;
        termType: TermType;
        definition: string;
    }
    interface KeyTerms {
        keyTerms: KeyTerm[];
    }
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" "\n".join(
                [x.text for x in highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "What are the topics of the excerpts? Be specific and concise."
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
        "content": "Based on the excerpts and identified topics, what are the key terms to know to understand the excerpts? Be concise. Respond with json that conforms to the following Typescript interface:\n" + TYPESCRIPT_KEY_TERM_INTERFACE
    })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.75,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    # messages.append({
    #     "role": "assistant",
    #     "content": response.choices[0].message.content
    # })
    # messages.append({
    #     "role": "user",
    #     "content": "Please be more concise."
    # })
    # response = openai_client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=.25,
    #     max_tokens=1000,
    #     top_p=1.0,
    #     frequency_penalty=.45,
    #     presence_penalty=.45,
    # )
    # print(response.choices[0].message.content)
    try:
        term_definition_json_str = extract_json(response.choices[0].message.content)
        term_definition_dict = json.loads(term_definition_json_str)
        CODE_TERM_TYPES = ['API_COMMAND', 'API_CLASS', 'API_METHOD']
        next_flashcards_json_dict = {
            "flashcards": [
                {
                    "id": uuid.uuid4(),
                    "question": '`' + term_definition['term'] + '`' if term_definition[
                                                                           'termType'] in CODE_TERM_TYPES else
                    term_definition['term'],
                    "answer": term_definition['definition']
                }
                for term_definition in term_definition_dict['keyTerms']
            ]
        }
        return next_flashcards_json_dict
    except Exception as e:
        print("error extracting json from response", e)
        # try one more time, noting the error
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": "Uh oh, looks like that wasn't quite valid json. Can you retry this, watch out for markdown syntax issues."
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
            term_definition_json_str = extract_json(response.choices[0].message.content)
            term_definition_dict = json.loads(term_definition_json_str)
            CODE_TERM_TYPES = ['API_COMMAND', 'API_CLASS', 'API_METHOD']
            next_flashcards_json_dict = {
                "flashcards": [
                    {
                        "question": '`' + term_definition['term'] + '`' if term_definition[
                                                                               'termType'] in CODE_TERM_TYPES else
                        term_definition['term'],
                        "answer": term_definition['definition']
                    }
                    for term_definition in term_definition_dict['keyTerms']
                ]
            }
            return next_flashcards_json_dict
        except Exception as e:
            print("error extracting json from response", e)
        return {"flashcards": []}


def make_flashcard_highlight_connections(good_flashcards_json_dict, highlights):
    # find the term/definition cards that don't have any highlights connected to them
    highlight_embeddings = get_embeddings([highlight.text for highlight in highlights])
    highlight_embeddings = sklearn.preprocessing.normalize(highlight_embeddings)
    db = get_direct_db()
    for flashcard in good_flashcards_json_dict['flashcards']:
        # find the highlight that this flashcard was generated from
        flashcard_embedding = get_embeddings([flashcard['question'] + "\n" + flashcard['answer']])
        flashcard_embedding = sklearn.preprocessing.normalize(flashcard_embedding)
        # identify three highlights with closest dot product to flashcard embedding
        closest_highlights_and_embedding = sorted(list(zip(highlights, highlight_embeddings)),
                                                  key=lambda x: np.dot(x[1], flashcard_embedding), reverse=True)[:3]
        for highlight, _ in closest_highlights_and_embedding:
            connection_model = FlashcardHighlightSource(
                flashcard_id=flashcard['id'],
                highlight_id=highlight.id,
            )
            db.add(connection_model)
    db.commit()
    db.close()


def generate_term_definition_flashcards_for_source(highlight_source_id):
    # get all the highlights for this source
    db = get_direct_db()
    highlight_source = db.query(HighlightSource).filter(HighlightSource.id == highlight_source_id).first()
    # get the most recent created_at field for term/definition cards from this source
    most_recent_execution_for_this_source = (db.query(sqlalchemy.func.max(Flashcard.created_at))
                                             .filter(Flashcard.highlight_source_id == highlight_source_id, Flashcard.generation_type == FlashcardGenerationType.TERM_DEFINITION_FLASHCARD.value).first())[0]
    highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source_id, Highlight.created_at >= most_recent_execution_for_this_source).all()
    db.close()
    flashcards_json_dict = {
        "flashcards": [],
    }
    for i in range(0, len(highlights), 12):
        next_highlights = highlights[i:i + 12]
        next_flashcards_json_dict = get_term_definition_flashcards_from_highlights(highlight_source, next_highlights)
        # filter out terms that are already present in the flashcards_json_dict
        next_flashcards_json_dict['flashcards'] = [flashcard for flashcard in next_flashcards_json_dict['flashcards'] if
                                                   flashcard['question'] not in [x['question'] for x in
                                                                                 flashcards_json_dict['flashcards']]]
        flashcards_json_dict['flashcards'] += next_flashcards_json_dict['flashcards']
        print("------")
        print("------")
    # now we need to filter out any flashcards that are sufficiently far away from the overall topics of the highlights
    good_flashcards_json_dict = get_relevant_term_definition_flashcards(flashcards_json_dict, highlight_source,
                                                                        highlights)
    insert_flashcard(good_flashcards_json_dict, highlight_source.readable_title,
                     FlashcardGenerationType.TERM_DEFINITION_FLASHCARD.value)
    make_flashcard_highlight_connections(good_flashcards_json_dict, highlights)


def get_highlights_not_used_in_flashcard_type(highlight_source_id: str, generation_type: FlashcardGenerationType):
    db = get_direct_db()
    highlights_that_have_cards_of_this_type_associated_with_them = set((db.query(Highlight)
                                                                        .join(FlashcardHighlightSource,
                                                                              Highlight.id == FlashcardHighlightSource.highlight_id)
                                                                        .join(Flashcard,
                                                                              FlashcardHighlightSource.flashcard_id == Flashcard.id)
                                                                        .filter(
        Highlight.source_id == highlight_source_id,
        Flashcard.generation_type == generation_type.value).distinct().all()))
    highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source_id).all()
    db.close()
    highlights_not_used_in_flashcard_type = [highlight for highlight in highlights if
                                             highlight not in highlights_that_have_cards_of_this_type_associated_with_them]
    return highlights_not_used_in_flashcard_type


def add_generation_type_to_flashcards():
    db = get_direct_db()
    flashcards_missing_generation_type = db.query(Flashcard).filter(Flashcard.generation_type == None).all()
    for flashcard in flashcards_missing_generation_type:
        flashcard.generation_type = FlashcardGenerationType.BASIC_FRONT_BACK_FLASHCARD.value
    db.commit()
    db.close()


def add_source_id_to_flashcards(source_id):
    db = get_direct_db()
    flashcards_missing_source_id = db.query(Flashcard).filter(Flashcard.highlight_source_id == None).all()
    for flashcard in flashcards_missing_source_id:
        flashcard.highlight_source_id = source_id
    db.commit()
    db.close()


def main():
    # the flashcard generation process is fairly involved and involves a pipeline of several steps
    highlight_source_id = '0a16c1a1-33b3-4777-8d2a-59347d1a985a'
    # generate_basic_front_back_flashcards_for_source(highlight_source_id)
    # generate_term_definition_flashcards_for_source(highlight_source_id)

if __name__ == '__main__':
    main()
