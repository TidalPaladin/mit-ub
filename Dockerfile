FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install dev dependencies
RUN pip install pytest pytest-mock pytest-cov autoflake autopep8 black flake8 isort clang-format

# Install dependencies
RUN pip install pandas pytorch-lightning torchmetrics jsonargparse pytorch-optimizer

# Install deep-helpers
RUN pip install "git+https://github.com/TidalPaladin/deep-helpers.git"

# Define volumes
VOLUME /app/config
VOLUME /app/data
VOLUME /app/outputs

# Add sources
ADD mit_ub /app/mit_ub/
ADD csrc /app/csrc/
ADD tests /app/tests/
ADD pyproject.toml setup.py /app/
WORKDIR /app

# Build
RUN pip install . --no-deps --no-build-isolation
