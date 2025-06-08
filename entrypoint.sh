#!/bin/sh
# entrypoint.sh

# Export Flask environment
export FLASK_APP=app.py
export FLASK_ENV=production

# Start the Flask API in the background
flask run --host=0.0.0.0 --port=5000 &

# Then start the Streamlit dashboard in the foreground
streamlit run dashboard.py \
  --server.address=0.0.0.0 \
  --server.port=8501 