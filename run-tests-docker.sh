#!/bin/bash

docker compose down 

docker compose run --rm --entrypoint python cli -m pytest -q

docker compose down 
