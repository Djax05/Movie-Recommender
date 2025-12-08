from fastapi import APIRouter, HTTPException, Query
from typing import List
from .models import (
    RecommendationRequest,
    RecommendationResponse,
    MovieRecommendation,
    HealthResponse
    )

router = APIRouter()

_recommend = None


def set_recommender(recommender):
    global _recommend
    _recommend = recommender


def get_recommender():
    if _recommend is None:
        raise RuntimeError("Recommender not initilaized")
    return _recommend


@router.get("/", response_model=HealthResponse)
def health_check():
    recommender = get_recommender()
    num_of_movies = recommender.get_total_movies()
    return HealthResponse(
        status="healthy",
        message="Movie Recommender is running " +
        f"with {num_of_movies} movies.",
        version="1.0.0"
    )


@router.post("/recommend", response_model=RecommendationResponse)
def recommend_movies(request: RecommendationRequest):
    recommender = get_recommender()

    results = recommender.get_recommendations(
        title=request.title,
        k=request.num_recommendations
    )

    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{request.title} not found in database." +
            "Try searching for similar titles."
        )

    recommendations = [
        MovieRecommendation(title=rec["title"], distance=rec["distance"])
        for rec in results
    ]

    return RecommendationResponse(
        query=request.title,
        recommendations=recommendations,
        count=len(recommendations)
    )


@router.get("/search", response_model=List[str])
def search_movies(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum Number"),
):
    recommender = get_recommender()
    matches = recommender.search_movies(query, limit)
    return matches


@router.get("/stats")
def get_stats():

    recommender = get_recommender()
    return {
        "total_movies": recommender.get_total_movies(),
        "weights": {
            "sentence": recommender.w_sentence,
            "genre": recommender.w_genre,
            "numeric": recommender.w_numeric,
        },
        "index_type": "PyNNDescent ANN"
    }
