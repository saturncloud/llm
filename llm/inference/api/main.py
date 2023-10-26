import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from llm.inference.multiproc import MultiprocessEngine
from llm.inference.types import InferenceRequest
from llm.model_configs import ModelConfig, VicunaConfig


engine: MultiprocessEngine
router = APIRouter()


@dataclass
class InferenceAPIRequest(InferenceRequest):
    stream: bool = False


@router.post("/api/inference")
def inference_endpoint(data: InferenceAPIRequest):
    stream = engine.add_request(data)

    # Stream partial responses as Server Sent Events
    if data.stream:
        def stream_resp():
            while True:
                response = stream.get()
                yield f"data: {json.dumps(response.to_dict())}\n\n"

                if response.stopped:
                    break
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_resp(), media_type="text/event-stream")

    # Wait for completion before sending response
    while True:
        response = stream.get()
        if response.stopped:
            return response.to_dict()


@asynccontextmanager
async def engine_lifecycle(app: FastAPI):
    yield
    engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "-m", "--model-id", help="Model ID", default=VicunaConfig.model_id
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        help="Number of chat models to run. Defaults to num GPUs.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        help="Maximum batch size for a single model",
        default=8,
    )
    parser.add_argument(
        "-d",
        "--delay",
        help="Maximum delay in seconfds after first request for batching",
        default=0.5,
    )
    parser.add_argument(
        "-p",
        "--pending",
        help="Maximum number of pending requests (does not include active)",
        default=-1.
    )
    parser.add_argument(
        "-q",
        "--quantization",
        choices=("4bit", "8bit"),
        help="Quantize the model during load with bitsandbytes",
    )
    args = parser.parse_args()

    quantization = args.quantization.lower() if args.quantization else None
    config = ModelConfig.from_registry(args.model_id)
    engine = MultiprocessEngine.from_model_config(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_delay=args.delay,
        max_pending=args.pending,
        load_kwargs={
            "quantization": quantization if quantization else False
        }
    )

    app = FastAPI(lifespan=engine_lifecycle, engine=engine)
    app.include_router(router)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
