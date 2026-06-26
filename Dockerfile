FROM python:3.11-slim

WORKDIR /opt/virufunc-atlas
COPY benchmark ./benchmark
COPY examples ./examples

RUN pip install --no-cache-dir numpy scikit-learn pandas pyyaml

CMD ["python", "benchmark/evaluate_predictions.py", "--help"]
