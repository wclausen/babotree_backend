import datetime
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Union, Dict, List, Optional, Tuple

import markdown
import openai
import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware

from app.babotree_utils import get_secret
from app.database import get_db
from app.models import Highlight, HighlightSource, HighlightSourceOutline, Flashcard, ContentEmbedding, SourceType, \
    FlashcardHighlightSource

app = FastAPI()

origins = [
    "https://babotree.com",
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

class ApiFlashcard(BaseModel):
    id: uuid.UUID
    topic: str
    question: str
    answer: str

@app.get("/flashcards")
def get_main_flashcards(db: Session = Depends(get_db)):
    """
    Returns the flashcards for the main page
    """
    flashcards = db.query(Flashcard).order_by(Flashcard.created_at.desc()).all()
    return {
        "flashcards": [ApiFlashcard(
        id=flashcard.id,
        topic=flashcard.topic,
        question=flashcard.question,
        answer=flashcard.answer) for flashcard in flashcards]}


class ApiRelatedHighlights(BaseModel):
    related_highlights: List[str]


@app.get("/flashcard/{flashcard_id}/related_highlights")
def get_related_highlights(flashcard_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Returns the related highlights for the given flashcard
    """
    flashcard = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
    if not flashcard:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    # check if we related highlights in the `flashcard_highlight_sources` table
    related_highlights = (db.query(Highlight)
    .join(FlashcardHighlightSource, FlashcardHighlightSource.highlight_id == Highlight.id)
                             .filter(FlashcardHighlightSource.flashcard_id == flashcard_id).all())
    if len(related_highlights) > 0:
        return ApiRelatedHighlights(
            related_highlights=[highlight.text for highlight in related_highlights]
        )
    # otherwise, we try the embeddings approach...
    print("Trying to get related highlights via embeddings")
    flashcard_embedding = db.query(ContentEmbedding).filter(ContentEmbedding.source_id == flashcard_id).first()
    if not flashcard_embedding:
        raise HTTPException(status_code=404, detail="No related highlights found")
    # get the related highlights, based on proximity to embedding
    closest_highlight_embeddings = (db.query(ContentEmbedding)
                                    .filter(ContentEmbedding.source_type == SourceType.HIGHLIGHT_TEXT.value)
                                    .order_by(
        ContentEmbedding.embedding.cosine_distance(flashcard_embedding.embedding)).limit(3).all())
    # get the actual highlights
    closest_highlight_ids = [embedding.source_id for embedding in closest_highlight_embeddings]
    related_highlights = db.query(Highlight).filter(Highlight.id.in_(closest_highlight_ids)).all()
    return ApiRelatedHighlights(
        related_highlights=[highlight.text for highlight in related_highlights]
    )


class ApiHighlight(BaseModel):
    id: uuid.UUID
    text: str


class ApiSourceHighlights(BaseModel):
    id: uuid.UUID
    book_title: str
    source_url: Optional[str]
    thumbnail_url: Optional[str]
    highlights: List[ApiHighlight]


class UserHighlights(BaseModel):
    highlights_by_source: List[ApiSourceHighlights]


def get_highlights_by_source(relevant_highlights: List[Tuple[Highlight, HighlightSource]]):
    """
    Groups the highlight objects by book, and maps the json data to proper pydantic models

    :param readwise_books_response:
    :param readwise_highlights_response:
    :return:
    """
    source_id_to_source = {}
    for _, source in relevant_highlights:
        if source.id not in source_id_to_source:
            source = ApiSourceHighlights(
                id=source.id,
                book_title=source.title,
                source_url=source.source_url,
                thumbnail_url=source.cover_image_url,
                highlights=[],  # we'll fill this in later
            )
            source_id_to_source[source.id] = source
    for highlight, _ in relevant_highlights:
        source_id = highlight.source_id
        if source_id not in source_id_to_source:
            continue
        source = source_id_to_source[source_id]
        source.highlights.append(ApiHighlight(
            id=highlight.id,
            text=highlight.text,
        ))
    for source in source_id_to_source.values():
        source.highlights.reverse()
    return list(source_id_to_source.values())


def _get_readwise_highlights(db: Session):
    fifteen_days_ago = datetime.datetime.now() - datetime.timedelta(days=15)
    relevant_highlights_and_sources = (db.query(Highlight, HighlightSource)
                            .outerjoin(HighlightSource, HighlightSource.id == Highlight.source_id)
                           .filter(Highlight.created_at > fifteen_days_ago)
                            .order_by(Highlight.created_at.desc())
                                       .all())
    return UserHighlights(
        highlights_by_source=get_highlights_by_source(relevant_highlights_and_sources)
    )

@app.get("/readwise")
def get_readwise_highlights(db: Session = Depends(get_db)):
    start_time = time.time()
    result = _get_readwise_highlights(db)
    end_time = time.time()
    print(f"Time to fetch readwise highlights: {end_time - start_time}")
    return result

@app.get("/similar_highlights")
def get_similar_highlights(hids: List[str], db: Session = Depends(get_db)):
    start_time = time.time()
    # fetch the highlights from the db
    highlights = db.query(Highlight).filter(Highlight.id.in_(hids)).all()
    # fetch the embeddings from the db
    embeddings = db.query(ContentEmbedding).filter(ContentEmbedding.source_id.in_(hids)).all()


openai.api_key = get_secret('OPENAI_API_KEY')


class GenerateQuestionsRequest(BaseModel):
    highlight_ids: List[uuid.UUID]

class QuestionAnswerPair(BaseModel):
    id: uuid.UUID
    question: str
    answer: str


class GenerateQuestionsResponse(BaseModel):
    questions_and_answers: List[QuestionAnswerPair]


def get_highlight_detail(highlight_id: str) -> ApiHighlight:
    print(f"Requesting highlight detail for: {highlight_id}")
    readwise_api_key = get_secret('READWISE_ACCESS_TOKEN')
    headers = {
        'Authorization': f'Token {readwise_api_key}',
    }
    response = requests.get(f'https://readwise.io/api/v2/highlights/{highlight_id}/',
                            headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch highlight detail")
    return ApiHighlight(
        id=highlight_id,
        text=response.json()['text']
    )



@app.post("/generate_questions")
def generate_questions(generate_questions_request: GenerateQuestionsRequest, db: Session = Depends(get_db)):
    print(generate_questions_request.highlight_ids)
    # fetch highlight ids from readwise
    results = db.query(Highlight).filter(Highlight.id.in_(generate_questions_request.highlight_ids)).all()
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

def _get_source_outline(source_id: uuid.UUID, db: Session):
    highlight_source_outline = db.query(HighlightSourceOutline).filter(HighlightSourceOutline.source_id == source_id).first()
    return {
        "outline_md": highlight_source_outline.outline_md,
        "outline_html": markdown.markdown(highlight_source_outline.outline_md)
    }

@app.get("/source/outline/{source_id}")
def get_source_outline(source_id: uuid.UUID, db: Session = Depends(get_db)):
    start_time = time.time()
    result = _get_source_outline(source_id, db)
    end_time = time.time()
    print(f"Time to fetch source outline: {end_time - start_time}")
    return result
