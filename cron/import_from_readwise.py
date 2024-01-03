import datetime
import uuid

import requests

from app.babotree_utils import get_secret
from app.database import get_direct_db
from app.models import Highlight, HighlightSource


def get_last_run_date():
    db = get_direct_db()

def get_highlight_data(last_updated_date, page_cursor=None):
    readwise_api_key = get_secret('READWISE_ACCESS_TOKEN')
    headers = {
        'Authorization': f'Token {readwise_api_key}',
    }
    query_params = ''
    if last_updated_date:
        query_params += f'updatedAfter={last_updated_date}'
    if page_cursor:
        query_params += f'&pageCursor={page_cursor}'
    response = requests.get(f'https://readwise.io/api/v2/export?{query_params}',
                            headers=headers)
    book_data = []
    highlight_data = []
    for result in response.json()['results']:
        book_data.append(result)
        for highlight in result['highlights']:
            highlight_data.append(highlight)
    return highlight_data, book_data, response.json()['nextPageCursor']

def to_highlight_source_db_model(api_book_dict):
    return HighlightSource(
        id=uuid.uuid4(),
        readwise_id=api_book_dict['user_book_id'],
        title=api_book_dict['title'],
        readable_title=api_book_dict['readable_title'],
        author=api_book_dict['author'],
        source=api_book_dict['source'],
        cover_image_url=api_book_dict['cover_image_url'],
        unique_url=api_book_dict['unique_url'],
        category=api_book_dict['category'],
        readwise_url=api_book_dict['readwise_url'],
        source_url=api_book_dict['source_url'],
        asin=api_book_dict['asin'],
    )
def to_highlight_db_model(highlight, highlight_source_id):
    return Highlight(
        readwise_id=highlight['id'],
        text=highlight['text'],
        source_id=highlight_source_id,
        source_location=highlight['location'],
        source_end_location=highlight['end_location'],
        source_location_type=highlight['location_type'],
        note=highlight['note'],
        color=highlight['color'],
        highlighted_at=highlight['highlighted_at'],
        created_at=highlight['created_at'],
        updated_at=highlight['updated_at'],
        external_id=highlight['external_id'],
        url=highlight['url'],
        is_favorite=highlight['is_favorite'],
        is_discarded=highlight['is_discard'],
        readwise_url=highlight['readwise_url'],
    )


def get_existing_readwise_sources_map():
    db = get_direct_db()
    existing_readwise_id_to_highlight_source_map = {}
    for highlight_source in db.query(HighlightSource).all():
        existing_readwise_id_to_highlight_source_map[highlight_source.readwise_id] = highlight_source.id
    db.close()
    return existing_readwise_id_to_highlight_source_map

def get_existing_readwise_highlight_ids():
    db = get_direct_db()
    result = [x.readwise_id for x in db.query(Highlight.readwise_id).all()]
    db.close()
    return result



def main():
    last_updated_date = datetime.datetime.utcnow() - datetime.timedelta(days=2000)
    highlight_data, book_data, next_cursor = get_highlight_data(last_updated_date=last_updated_date)
    readwise_source_id_to_my_source_id = get_existing_readwise_sources_map()
    existing_readwise_higlight_ids = get_existing_readwise_highlight_ids()
    highlight_id_to_source_id = {}
    highlight_ids_added = set()
    db = get_direct_db()
    for book in book_data:
        if book['user_book_id'] not in readwise_source_id_to_my_source_id:
            db_model = to_highlight_source_db_model(book)
            readwise_source_id_to_my_source_id[book['user_book_id']] = db_model.id
            db.add(db_model)
        source_id = readwise_source_id_to_my_source_id[book['user_book_id']]
        for highlight in book['highlights']:
            highlight_id_to_source_id[highlight['id']] = source_id
    db.commit()
    for highlight in highlight_data:
        if highlight['id'] in existing_readwise_higlight_ids:
            # we've added this highlight through some other means
            continue
        highlight_source_id = highlight_id_to_source_id[highlight['id']]
        db_model = to_highlight_db_model(highlight, highlight_source_id)
        db.add(db_model)
        highlight_ids_added.add(highlight['id'])
    db.commit()
    while next_cursor:
        highlight_data, book_data, next_cursor = get_highlight_data(last_updated_date=last_updated_date, page_cursor=next_cursor)
        for book in book_data:
            if book['user_book_id'] not in readwise_source_id_to_my_source_id:
                db_model = to_highlight_source_db_model(book)
                readwise_source_id_to_my_source_id[book['user_book_id']] = db_model.id
                db.add(db_model)
            source_id = readwise_source_id_to_my_source_id[book['user_book_id']]
            for highlight in book['highlights']:
                highlight_id_to_source_id[highlight['id']] = source_id
        db.commit()
        for highlight in highlight_data:
            if highlight['id'] in existing_readwise_higlight_ids:
                # we've added this highlight through some other means
                continue
            highlight_source_id = highlight_id_to_source_id[highlight['id']]
            db_model = to_highlight_db_model(highlight, highlight_source_id)
            db.add(db_model)
        db.commit()
    db.close()
    print(f"Added {len(highlight_ids_added)} highlights")

if __name__ == '__main__':
    main()