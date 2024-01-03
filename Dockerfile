FROM python:3.9.16

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/main.py

COPY ./.env /code/.env

COPY ./app /code/app

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--proxy-headers", "--port", "10000"]