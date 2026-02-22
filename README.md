# Movie Recommender API

A content-based movie recommendation system built with FastAPI and Approximate Nearest Neighbors (ANN) search. This service uses sentence embeddings, genre encoding, and numeric features to provide highly relevant movie recommendations.

## Features

- **Content-Based Recommendations**: Uses a weighted combination of sentence embeddings, genre information, and numeric features
- **Fast ANN Search**: Leverages PyNNDescent for efficient approximate nearest neighbor queries
- **RESTful API**: Built with FastAPI for high performance and automatic interactive documentation
- **Movie Search**: Search for movies by partial title match
- **Health Checks**: Monitor API status and dataset statistics
- **CORS Support**: Enabled for cross-origin requests
- **Request Logging**: Automatic logging of all HTTP requests with response times
- **Docker Support**: Containerized for easy deployment

## Architecture

The system uses a multi-feature approach to generate recommendations:

- **Sentence Embeddings (50% weight)**: Captures semantic similarity between movie descriptions using transformer models
- **Genre Features (30% weight)**: Encodes movie genre information to ensure genre relevance
- **Numeric Features (20% weight)**: Incorporates quantifiable attributes like ratings, year, runtime, etc.

All features are combined using a weighted scoring mechanism and indexed with PyNNDescent for fast retrieval.

## Prerequisites

- Python >= 3.12
- pip or uv package manager
- (Optional) Docker for containerized deployment

## Installation

### Using Virtual Environment

1. Clone the repository and navigate to the project directory:
```bash
cd "Movie recommender"
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

Or with `uv` (faster):
```bash
uv sync
```

### Using Docker

Build and run the application in a Docker container:

```bash
docker build -t movie-recommender-api .
docker run -p 8000:8000 movie-recommender-api
```

## Running the Application

### Development Server

Start the FastAPI development server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Interactive Documentation

- **Swagger UI**: Visit `http://localhost:8000/docs`
- **ReDoc**: Visit `http://localhost:8000/redoc`

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API and total number of movies in the database.

**Response:**
```json
{
  "status": "healthy",
  "message": "Movie Recommender is running with 4764 movies.",
  "version": "1.0.0"
}
```

### Get Recommendations
```
POST /recommend
```
Get movie recommendations based on a given movie title.

**Request Body:**
```json
{
  "title": "The Matrix",
  "num_recommendations": 10
}
```

**Query Parameters:**
- `title` (string, required): Movie title to get recommendations for (min length: 1)
- `num_recommendations` (integer, optional): Number of similar movies to return (default: 10, range: 1-50)

**Response:**
```json
{
  "query": "The Matrix",
  "recommendations": [
    {
      "title": "The Matrix Reloaded",
      "distance": 0.12
    },
    {
      "title": "Equilibrium",
      "distance": 0.24
    }
  ],
  "count": 2
}
```

**Error Response (404):**
- Returns a 404 error if the movie title is not found in the database

### Search Movies
```
GET /search
```
Search for movies by partial title match.

**Query Parameters:**
- `query` (string, required): Search query (min length: 1)
- `limit` (integer, optional): Maximum number of results to return (default: 10, range: 1-50)

**Response:**
```json
[
  "The Matrix",
  "The Matrix Reloaded",
  "The Matrix Revolutions"
]
```

### Get Statistics
```
GET /stats
```
Retrieve API statistics including dataset size and feature weights.

**Response:**
```json
{
  "total_movies": 123456,
  "weights": {
    "sentence": 0.5,
    "genre": 0.3,
    "numeric": 0.2
  },
  "index_type": "PyNNDescent ANN"
}
```

## Project Structure

```
.
├── api/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                  # FastAPI app setup and middleware
│   ├── router.py                # API endpoints
│   ├── models.py                # Pydantic request/response models
│   ├── config.py                # Configuration settings
│   └── core/
│       └── logging.py           # Logging setup
├── recommender/                  # Recommendation engine
│   ├── recommender.py           # Main recommender class
│   ├── build_index.py           # Index building utilities
│   └── text_cleaner.py          # Text preprocessing utilities
├── data/                         # Data directory
│   ├── raw/                     # Raw input data
│   ├── processed/               # Processed data and embeddings
│   │   ├── sentence_embeddings.npy    # sentence transformer embeddings
│   │   ├── numeric_scaled.npy         # normalized numeric features
│   │   └── encoded_data.csv           # encoded genre features
│   └── index/                   # Pre-built indexes
│       ├── pynndescent_index.pkl      # PyNNDescent ANN index
│       └── title_to_index.json        # Movie title to index mapping
├── notebooks/                    # Jupyter notebooks
│   ├── data_analysis.ipynb      # EDA and data exploration
│   ├── model.ipynb              # Model training and evaluation
│   └── visualization.ipynb      # Results visualization
├── tests/                        # Testing utilities
│   └── evaluate_ann.py          # ANN index evaluation
├── pyproject.toml               # Project dependencies and metadata
├── Dockerfile                   # Docker configuration
└── README.md                    # This file
```

## Configuration

The API can be configured via environment variables using a `.env` file or through direct settings. Key configuration options:

```python
# Server
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# API Information
API_TITLE = "Movie recommendation API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Content based movie recommendation using ANN"

# Recommendation Settings
MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 50
DEFAULT_RECOMMENDATIONS = 10

# Feature Weights
SENTENCE_WEIGHT = 0.5  # Semantic similarity
GENRE_WEIGHT = 0.3    # Genre relevance
NUMERIC_WEIGHT = 0.2  # Numeric attributes
```

## Dependencies

Key dependencies used in this project:

- **fastapi** (>=0.121.3): Modern web framework for building APIs
- **pydantic** (>=2.12.4): Data validation using Python type annotations
- **uvicorn** (>=0.38.0): ASGI web server
- **numpy** (>=2.3.4): Numerical computing
- **pandas** (>=2.3.3): Data manipulation
- **scikit-learn** (>=1.7.2): ML preprocessing and utilities
- **sentence-transformers** (>=5.1.2): Semantic embeddings
- **pynndescent** (>=0.5.13): Approximate nearest neighbor search
- **scipy** (>=1.16.3): Scientific computing

See [pyproject.toml](pyproject.toml) for the complete list of dependencies.

## Development

### Notebooks

The project includes Jupyter notebooks for data exploration and model development:

- **data_analysis.ipynb**: Exploratory data analysis of movie metadata
- **model.ipynb**: Model training, feature extraction, and index building
- **visualization.ipynb**: Visualization of recommendations and embeddings

Run notebooks with:
```bash
jupyter notebook notebooks/
```

### Testing

Evaluate the ANN index performance:
```bash
python tests/evaluate_ann.py
```

## Performance Notes

- The recommender uses PyNNDescent for efficient approximate nearest neighbor search
- Sentence embeddings are pre-computed using sentence-transformers for fast inference
- Feature scaling is performed using scikit-learn's scalers for consistent recommendations
- CORS is enabled for all origins (configurable in production)

## Deployment

The application includes a Dockerfile for containerized deployment:

```bash
# Build
docker build -t movie-recommender .

# Run
docker run -p 8000:8000 movie-recommender

# Health check
curl http://localhost:8000/health
```

The Docker image includes:
- Python 3.12-slim base image
- System dependencies for ML libraries
- optimized layer caching for faster builds
- Health checks for monitoring

## License

[Add your license information here]

## Author

[Add author information here]
