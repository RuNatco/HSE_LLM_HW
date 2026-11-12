FROM vllm/vllm-openai:v0.22.0-cu129@sha256:14f7be59bea024f60a7cb7076ce5666d7c72751354a3db22cb713af28cffa2cf

RUN apt-get purge -y cuda-compat-12-9
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0" HF_HOME=/cache \
    VLLM_CACHE_ROOT=/cache/vllm TRITON_CACHE_DIR=/cache/triton

RUN printf 'numpy==2.2.6\n' | uv pip install --system --no-cache --override /dev/stdin \
    vllm==0.22.0 trl==1.16.0 transformers==5.9.0 peft==0.18.1 \
    datasets==4.7.0 accelerate==1.10.1 flash-linear-attention==0.4.1 \
    bfcl-eval==2026.3.23 soundfile==0.14.0 scipy==1.16.2

RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 \
    uv pip install --system --no-cache --no-build-isolation causal-conv1d==1.7.0

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
