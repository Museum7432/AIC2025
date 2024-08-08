# Using miniconda as base image
FROM continuumio/miniconda3

# Setting working directory
WORKDIR /app
# Copying the environment.yml file to the image
COPY ./env.yaml .
# Dealing with CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/linux-64/current_repodata.json>
RUN conda config --set ssl_verify no
# Building the environment
# RUN conda env create -f environment.yml
RUN conda env update -f env.yaml
RUN rm env.yaml
# Refer to https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "semantic", "/bin/bash", "-c"]
# # Exposing port 8000
# EXPOSE 8000

# # Running uvicorn server
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "semantic", \
#     "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]