FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN chmod 1777 /tmp && \
    apt-get update && apt-get install -y \
        python3.10 python3-dev git wget curl build-essential uuid-runtime \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Node.js 22.x
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# vLLM + ML packages (matching PostTrainBench standard.def)
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=auto
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
