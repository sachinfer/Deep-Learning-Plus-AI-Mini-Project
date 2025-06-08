#!/bin/sh
# entrypoint.sh

# Export Flask environment
export FLASK_APP=app.py
export FLASK_ENV=production

# Start the Flask API with Gunicorn in the background
gunicorn -w 4 -b 0.0.0.0:5000 app:app &

# Then start the Streamlit dashboard in the foreground
streamlit run dashboard.py \
  --server.address=0.0.0.0 \
  --server.port=8501 