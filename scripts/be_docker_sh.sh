docker run -it \
  --name abc \
  -p 8000:8000 \
  -v "$(pwd)":/app \
  -v /home/toannk/.cache/huggingface:/root/.cache/huggingface \
  -v /home/toannk/.cache/torch:/root/.cache/torch \
  aic_be \
  conda run --no-capture-output -n semantic uvicorn api.main:app --host 0.0.0.0 --port 8000