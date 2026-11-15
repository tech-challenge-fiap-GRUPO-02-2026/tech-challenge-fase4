#!/bin/bash

docker compose down 

docker compose run --rm cli run-all

docker compose down 