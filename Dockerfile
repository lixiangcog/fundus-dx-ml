FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./best_model.pth /code/best_model.pth
COPY ./api /code/api
COPY ./shared.py /code/shared.py

# Non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# api/main.py loads best_model.pth via a relative path, so stay in /code
WORKDIR /code

# Hugging Face Spaces default port
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
