# Use Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade
RUN pip install --upgrade pip

# Copy files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Download spaCy model and link
RUN python -m spacy download https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.0/en_core_web_md-3.7.0.tar.gz \
 && python -m spacy link en_core_web_md en_core_web_md

# Expose port (adjust to your app if needed)
EXPOSE 10000

# Start the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
