FROM python:3.10-slim

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /code
COPY --chown=user:user . .

# Install uv, create virtual environment, and install dependencies
RUN pip install --no-cache-dir uv
RUN uv venv .venv
ENV PATH="/code/.venv/bin:$PATH"
RUN uv pip install -r requirements.txt

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
