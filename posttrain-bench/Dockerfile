FROM vllm/vllm-openai:v0.11.0 AS base

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN chmod 1777 /tmp && \
    apt-get update && apt-get install -y \
        git wget curl build-essential uuid-runtime \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22.x
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ML packages on top of vLLM base
RUN uv pip install --system --no-cache ninja packaging
RUN uv pip install --system --no-cache \
    accelerate boto3 bitsandbytes datasets evaluate lm-eval openai \
    pandas scikit-learn shortuuid tokenizers transformers trl peft \
    tiktoken inspect-ai matplotlib certifi
RUN uv pip install --system --no-cache flash_attn --no-build-isolation

# inspect_evals
RUN cd /opt && \
    git clone --depth=1 https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd inspect_evals && uv pip install --system --no-cache .

ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"
