import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from recommender.recommender import MovieRecommender

from .config import settings
from .router import router, set_recommender
from .core.logging import setup_logging, get_logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print(" Staring Movie Recommender API...")
    print("="*50)

    recommender = MovieRecommender()
    set_recommender(recommender)

    print("="*50)
    logger.info("API ready to serve requests!")

    yield

    print("\n Shutting Down API...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} | "
        f"{response.status_code} | {duration:.2f}ms"
    )

    return response


app.include_router(router, tags=["recommendations"])


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Hello from movie-recommender!",
        "docs": "/docs",
        "health": "/health"
        }

# def main():
#     print("Hello from movie recommender")


# if __name__ == "__main__":
#     main()
