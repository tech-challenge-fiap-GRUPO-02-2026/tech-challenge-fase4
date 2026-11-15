#!/bin/bash

docker compose down

# --service-ports publica 127.0.0.1:8766 (o "run" ignora as portas por padrão)
docker compose run --rm --service-ports painel

