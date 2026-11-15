#!/bin/bash


docker compose down 

docker compose run --rm cli generate-data

docker compose down 