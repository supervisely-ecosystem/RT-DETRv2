FROM supervisely/rt-detrv2:1.0.3

WORKDIR /app
COPY . /app

EXPOSE 80

ENV PYTHONPATH "${PYTHONPATH}:/app/supervisely_integration/serve"
ENV APP_MODE=production ENV=production

ENTRYPOINT ["python3", "supervisely_integration/serve/main.py"]
# CMD ["--model", "RT-DETRv2-S", "--predict", "demo/image.jpg"]
