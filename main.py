import datetime
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Union, Dict, List, Optional

import openai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from babotree_utils import get_secret

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"]
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Highlight(BaseModel):
    id: int
    text: str


class BookHighlights(BaseModel):
    id: int
    book_title: str
    source_url: Optional[str]
    thumbnail_url: Optional[str]
    highlights: List[Highlight]


class UserHighlights(BaseModel):
    highlights_by_source: List[BookHighlights]


def get_highlights_by_source(readwise_books, readwise_highlights):
    """
    Groups the highlight objects by book, and maps the json data to proper pydantic models

    :param readwise_books_response:
    :param readwise_highlights_response:
    :return:
    """
    book_id_to_book = {}
    for obj in readwise_books:
        book = BookHighlights(
            id=obj['id'],
            book_title=obj['title'],
            source_url=obj['source_url'],
            thumbnail_url=obj['cover_image_url'],
            highlights=[],  # we'll fill this in later
        )
        if obj['id'] not in book_id_to_book:
            book_id_to_book[obj['id']] = book
    for obj in readwise_highlights:
        book_id = obj['book_id']
        if book_id not in book_id_to_book:
            continue
        book = book_id_to_book[book_id]
        book.highlights.append(Highlight(
            id=obj['id'],
            text=obj['text'],
        ))
    for book in book_id_to_book.values():
        book.highlights.reverse()
    return list(book_id_to_book.values())


@app.get("/readwise")
def get_readwise_highlights():
    readwise_api_key = get_secret('READWISE_ACCESS_TOKEN')
    headers = {
        'Authorization': f'Token {readwise_api_key}',
    }
    yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=15)
    readwise_highlights_response = requests.get(
        f'https://readwise.io/api/v2/highlights/?page_count=1000&updated__gt={yesterday}', headers=headers)
    readwise_highlights = readwise_highlights_response.json()['results']
    print(readwise_highlights[:2])
    readwise_books_response = requests.get(f'https://readwise.io/api/v2/books/?page_count=1000&updated__gt={yesterday}', headers=headers)
    readwise_books = readwise_books_response.json()['results']
    print()
    print(readwise_books[:2])

    # print(readwise_books)
    # print(readwise_articles_response.json()['results'][:2])
    return UserHighlights(
        highlights_by_source=get_highlights_by_source(readwise_books, readwise_highlights)
    )

openai.api_key = get_secret('OPENAI_API_KEY')


class GenerateQuestionsRequest(BaseModel):
    highlight_ids: List[int]

class QuestionAnswerPair(BaseModel):
    id: uuid.UUID
    question: str
    answer: str


class GenerateQuestionsResponse(BaseModel):
    questions_and_answers: List[QuestionAnswerPair]


def get_highlight_detail(highlight_id: str) -> Highlight:
    print(f"Requesting highlight detail for: {highlight_id}")
    readwise_api_key = get_secret('READWISE_ACCESS_TOKEN')
    headers = {
        'Authorization': f'Token {readwise_api_key}',
    }
    response = requests.get(f'https://readwise.io/api/v2/highlights/{highlight_id}/',
                            headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch highlight detail")
    return Highlight(
        id=highlight_id,
        text=response.json()['text']
    )



@app.post("/generate_questions")
def generate_questions(generate_questions_request: GenerateQuestionsRequest):
    print(generate_questions_request.highlight_ids)
    # fetch highlight ids from readwise
    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(generate_questions_request.highlight_ids))) as executor:
        # readwise api requires requesting highlight details individually
        futures = []
        for highlight_id in generate_questions_request.highlight_ids:
            futures.append(executor.submit(get_highlight_detail, highlight_id))
        for future in futures:
            try:
                results.append(future.result())
            except HTTPException as e:
                raise e
    # now we need to send the highlight data to openai to get questions related to these highlights
    openai_question_function_tools = [{
        "type": "function",
        "function": {
            "name": "create_question_answer_pair",
            "description": "Creates a question answer pair",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question string",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The correct answer to the question",
                    },
                },
                "required": ["question", "answer"]
            },
        }
    },
    ]
    messages = [
        {
            "role": "system",
            "content": "You are an expert educator, a master of helping students traverse Bloom's taxonomy and understand subjects on a deep level.",
        },
        {
            "role": "user",
            "content": "Please create 3 question/answer pairs based on the following content, include which option is the correct answer:\n\n" + "\n".join(
                [highlight.text for highlight in results]),
        },
    ]
    print("Fetching questions from openai")
    response = openai.chat.completions.create(
        model='gpt-4-1106-preview',
        tools=openai_question_function_tools,
        # tool_choice={"type": "function", "function": {"name": "create_question_answer_pair"}},
        messages=messages,
        temperature=1.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.45,
    )
    print(response.choices[0].message.tool_calls)
    results: List[QuestionAnswerPair] = []
    # first, make sure we have some function call responses
    if response.choices[0].message.tool_calls:
        # extract the question/answer pairs from the tool_calls list items
        for tool_call in response.choices[0].message.tool_calls:
            # tool_call is a dict with the following fields, 'id', 'type', and 'function'
            # 'function' contains the information we want, which would be in the 'arguments' key in the 'function' dict
            if tool_call.function.arguments:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if 'question' in arguments and 'answer' in arguments:
                        results.append(QuestionAnswerPair(
                            id=uuid.uuid4(),
                            question=arguments['question'],
                            answer=arguments['answer']
                        ))
                    else:
                        print("Result from openai doesn't seem to contain proper params...")
                except JSONDecodeError:
                    print("Result from openai doesn't seem to be valid JSON...")
    return GenerateQuestionsResponse(
        questions_and_answers=results
    )




@app.post("/summarize")
def get_openai_summary(source_id: int):
    readwise_api_key = get_secret('READWISE_ACCESS_TOKEN')
    headers = {
        'Authorization': f'Token {readwise_api_key}',
    }
    source_highlights_response = requests.get(
        f'https://readwise.io/api/v2/highlights/?page_count=1000&book_id={source_id}', headers=headers)
    source_highlights = source_highlights_response.json()['results']
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Please create 3 multiple choice questions based on the following content, include which option is the correct answer:" + "\n".join([highlight['text'] for highlight in source_highlights]),
        },
    ]
    response = openai.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=messages,
        temperature=1.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.45,
    )
    print(response.choices[0].message.content)