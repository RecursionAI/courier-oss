# Courier OSS: Managed LLM Inference Engine

Courier OSS is a production-grade, API-controlled inference engine designed for hosting Large Language Models (LLMs) on
private hardware. It is specifically optimized for seamless integration with automation platforms like n8n and provides
a robust alternative to public AI APIs by allowing full control over model deployment, memory management, and LoRA
adapters.

## Key Features

- Dynamic Model Management: Automatically load and unload models based on demand and hardware constraints.
- VRAM Optimization: Intelligent LRU (Least Recently Used) policy for "Flex" models, ensuring your GPU memory is used
  efficiently.
- LoRA Support: Native support for loading Low-Rank Adaptation (LoRA) adapters on top of base models.
- Robust Inference: Easy-to-use API for text, vision, and audio tasks.
- Multi-Tenant Analytics: Track token usage, memory consumption, and request latency per API key.
- Production Ready: Built-in health monitoring, automatic cleanup of expired models, and background task processing.

## Architecture

Courier OSS sits between your application and the vLLM inference engine.

1. API Layer (FastAPI): Handles authentication, model registration, and inference requests.
2. Model Manager: The brain of the system. It tracks available VRAM and decides when to load new models or unload idle
   ones.
3. vLLM Model Pool: Wraps vLLM's AsyncLLMEngine for high-performance, hardware-aware inference.
4. Database (FlowDB): Persists model configurations, user permissions, and analytics.

## Requirements

- OS: Linux (recommended for CUDA support).
- GPU: NVIDIA GPU with CUDA support (Compute Capability 8.0+ recommended for Bfloat16). Drivers must be installed. Run
  `nvidia-smi` to verify.
- Python: 3.10+
- Dependencies: vLLM, FastAPI, PyTorch, FlowDB.

Note: ffmpeg is required for audio processing features.

## Quick Start

1. Clone the repo:
   ```bash
   git clone https://github.com/RecursionAI/courier-oss.git
   cd courier-oss
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup FlowDB:
   Courier OSS requires FlowDB as its database. Please clone FlowDB and follow its setup instructions:
   ```bash
   git clone https://github.com/RecursionAI/FlowDB.git
   ```
   Follow the FlowDB README for detailed configuration and installation.

4. Configure Environment:
   Create a .env file in the root directory:
   ```env
   ADMIN_KEY=your_secure_admin_key
   FLOWDB_API_KEY=your_flowdb_api_key
   FLOWDB_URL=your_flowdb_url
   TENSOR_PARALLEL_SIZE=1
   PIPELINE_PARALLEL_SIZE=1
   ```

NOTE: look at the [.env.example](.env.example) file for more help.

5. Run the Server:
   ```bash
   python -m src.endpoints
   ```

## API Documentation

FastAPI automatically generates Swagger docs. Once the server is running, visit `<host_url>/docs` for the interactive
Swagger UI.

All the APIs will be visible including their request bodies and you can make requests directly from the UI.

## Connecting to Courier Dashboard

Courier is an AI API platform. While it offers paid APIs, you can connect your own GPU resources **completely free**
using Courier OSS and manage your workbench remotely as well as gain access to the analytics dashboard.

### Connection Guide

1. Make sure courier-oss is running on your machine.
2. Create a web API (we recommend using **ngrok**)
3. Go to https://courier.thinkrecursion.ai/ and login or create an account (free)
4. In the top-center of the dashboard you will see `Courier Workspace` and a `+` button. Click on the `+` button to
   create a new workspace.
5. Enter a name for your workspace, your web API for the base URL, and your admin API key.
6. Courier will automatically connect to your courier-oss instance, and you can start using it right away.

## Contributing

We welcome contributions! Please see our Contributing Guidelines (CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache 2.0 LICENSE file included in this repository.
