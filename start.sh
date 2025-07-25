#!/bin/bash

# Install the spaCy model
python -m spacy download en_core_web_md

# Start the Flask app
gunicorn app:app --bind 0.0.0.0:$PORT
