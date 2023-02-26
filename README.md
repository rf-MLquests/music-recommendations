# music-recommendations

The core dataset is the Taste Profile Subset released by The Echo Nest as part of the Million Song
Dataset.

## Kind of Recommendation Techniques Included:

- Rank-based recommender
- Content-based recommender (tfidf features)
- Collaborative filtering recommenders

## Repository Content:

- FastAPI endpoints that exposes each recommender
- Dockerized for distribution and scaling

## Model-side TODOs:

Some may require the use of a different dataset

- Hybrid models
- Try other features for content-based models
- Transformer based models

## System-side TODOs:

- Mocking a production environment by generating new user-song interactions
- Adding backend storage to store above new data / updated dataset / models
- Adding model re-train components that periodically retrain models
- At inference time, retrieve latest stored model for fast prediction
