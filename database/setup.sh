#!/bin/sh
# set -x

# setup database parameters
DB_APP_NAME=${DB_APP_NAME:-predict-db}
DB_DATABASE="${DB_DATABASE:-predict-db}"
DB_USERNAME="${DB_USERNAME:-predict-db}"
DB_PASSWORD="${DB_PASSWORD:-failureislame}"

setup_container(){
  docker stop "${DB_APP_NAME}"

  docker run \
    --name "${DB_APP_NAME}" \
    -d --rm \
    -p 5432:5432 \
    -v $(pwd):/opt/app-root/src \
    -e POSTGRESQL_DATABASE="${DB_DATABASE}" \
    -e POSTGRESQL_PASSWORD="${DB_PASSWORD}" \
    -e POSTGRESQL_USER="${DB_USERNAME}" \
    registry.redhat.io/rhel8/postgresql-12:latest

  docker exec \
    -it \
    "${DB_APP_NAME}" \
    /bin/bash -c ". setup.sh; setup_db"
}

setup_db(){
  cp sensor.csv.zip db.sql /tmp

  cd /tmp

  unzip -o sensor.csv.zip
  
  echo 'GRANT ALL ON TABLE waterpump TO "'"${DB_USERNAME}"'" ;' >> db.sql
  psql -d "${DB_APP_NAME}" -f db.sql
}
