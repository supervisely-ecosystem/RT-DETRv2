FROM supervisely/rt-detrv2:1.0.3

WORKDIR /app
COPY . /app

EXPOSE 80

ENV PYTHONPATH "${PYTHONPATH}:/app/supervisely_integration/serve"
ENV APP_MODE=production ENV=production

ENTRYPOINT ["python3", "-u", "-m", "uvicorn", "supervisely_integration.serve.main:model.app"]

CMD ["--host", "0.0.0.0", "--port", "80"]
