# 1. Start with a base image that includes Python and the correct NVIDIA CUDA drivers
# This image is based on CUDA 11.8, matching your PyTorch install command.
FROM nvcr.io/nvidia/pytorch:23.09-py3

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system dependencies like ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

# 4. Copy and install your Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code into the container
COPY main.py ./

# 6. Expose the port the app will run on
EXPOSE 8080

# 7. Define the command to run your web application when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "main:app"]