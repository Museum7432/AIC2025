FROM continuumio/miniconda3

# Dealing with CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/linux-64/current_repodata.json>
RUN conda config --set ssl_verify no

COPY ./environment.yaml .

# Building the environment
RUN conda env create -f environment.yaml
# RUN conda env update -f environment.yaml

RUN rm environment.yaml