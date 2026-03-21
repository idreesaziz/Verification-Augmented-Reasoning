FROM python:3.12-slim

RUN pip install --no-cache-dir \
    numpy \
    sympy \
    scipy \
    z3-solver

WORKDIR /workspace

# Drop to non-root user
RUN useradd -m sandbox
USER sandbox

CMD ["python3", "-u", "-i"]
