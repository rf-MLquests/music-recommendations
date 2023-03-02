FROM jupyter/datascience-notebook

USER root

RUN mkdir music-recommendations

COPY . music-recommendations/

WORKDIR music-recommendations

RUN pip3 install -r requirements.txt

EXPOSE 8000

RUN python3 train.py