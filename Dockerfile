FROM continuumio/miniconda3:latest

# Dealing with CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/linux-64/current_repodata.json>
RUN conda config --set ssl_verify no

RUN apt-get update && apt-get install -y build-essential

COPY ./environment.yaml .

# Building the environment

RUN conda env create -f environment.yaml

# RUN conda install -y -n py311 openblas-devel -c anaconda
# RUN conda install -y -n py311 open-clip-torch=2.26.1
# RUN conda install -y -n py311 timm=1.0.9
# RUN conda env update -f environment.yaml

RUN rm environment.yaml

