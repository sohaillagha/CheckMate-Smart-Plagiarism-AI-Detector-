FROM python:3.10-slim

# Create a user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY --chown=user . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for running the Flask app
RUN pip install --no-cache-dir gunicorn

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose port 7860 (Required by Hugging Face)
EXPOSE 7860

# Command to run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "1200"]
