#!/bin/bash

docker compose down 

docker compose run --rm cli evaluate

docker compose down 