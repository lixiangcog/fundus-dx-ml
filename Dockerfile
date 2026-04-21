# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the model file
COPY ./best_model.pth /code/best_model.pth

# Copy the api directory
COPY ./api /code/api

# Copy shared utilities used by api/main.py
COPY ./shared.py /code/shared.py

# Create a user to run the application (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to /code/api so imports work as expected if they are relative
# However, the code in main.py does 'from api import ...' or similar? 
# Let's check main.py imports. It uses standard library and installed packages.
# But it loads "best_model.pth". 
# In main.py: model_path = "best_model.pth"
# So we need to be in the directory where best_model.pth is, OR update main.py to look in the right place.
# Let's set WORKDIR to /code so best_model.pth is in current dir.
WORKDIR /code

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the application
# We need to point to api.main:app. Since we are in /code, and api is a folder, it should be api.main:app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
