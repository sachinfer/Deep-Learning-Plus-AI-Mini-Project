# Use a slim Python image
FROM python:3.9-slim

# Set a working dir
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install libgomp1 for LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy your code & artifacts
COPY . .

# Expose both ports
EXPOSE 5000 8501

# Use our entrypoint to launch both services
ENTRYPOINT ["./entrypoint.sh"] 