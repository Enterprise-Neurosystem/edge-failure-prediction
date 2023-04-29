#!/bin/sh
# set -x

setup_container(){
  docker stop edge-db

  docker run \
    --name edge-db \
    -d --rm \
    -p 5432:5432 \
    -v $(pwd):/opt/app-root/src \
    -e POSTGRESQL_DATABASE=edge-db \
    -e POSTGRESQL_PASSWORD=failure \
    -e POSTGRESQL_USER=edge-db \
    registry.redhat.io/rhel8/postgresql-12:latest

  docker exec \
    -it \
    edge-db \
    /bin/bash -c ". setup.sh; setup_db"
}

setup_db(){
  cp sensor.csv.zip db.sql /tmp

  cd /tmp

  unzip -o sensor.csv.zip
  psql -d edge-db -f db.sql
}
