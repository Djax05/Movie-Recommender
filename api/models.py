from pydantic import BaseModel, Field
from typing import List


class RecommendationRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        description="Movie title to get recommendations for",
        examples=["The Matrix", "Inception", "The Dark Knight"]
    )

    num_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of similar movies to return"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "The Matrix",
                "num_recommendations": 10
            }
        }


class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    distance: float = Field(
        description="Similarity distance (lower = more similar)",
        ge=0.0
    )


class RecommendationResponse(BaseModel):
    """
    Example response:
    {
        "query": "The Matrix",
        "recommendations": [
            {"title": "The Matrix Reloaded", "distance": 0.12},
            {"title": "Equilibrum", "distance": 0.24}
            ],
            "count": 2
    }
    """
    query: str = Field(description="Original movie title searched")
    recommendations: List[MovieRecommendation] = Field(
        description="List of recommended movies, ordered by similarity"
    )
    count: int = Field(description="Number of recommendations returned")


class HealthResponse(BaseModel):

    status: str
    message: str
    version: str
