#!/bin/sh

# setup app parameters
APP_NAME="${APP_NAME:-predict}"
NAMESPACE="${NAMESPACE:-edge-failure-prediction}"
GIT_BRANCH="workshop/updates"

# setup database parameters
DB_APP_NAME=${DB_APP_NAME:-predict-db}
DB_HOSTNAME="${DB_HOSTNAME:-${APP_NAME}.${NAMESPACE}.svc.cluster.local}"
DB_DATABASE="${DB_DATABASE:-predict-db}"
DB_USERNAME="${DB_USERNAME:-predict-db}"
DB_PASSWORD="${DB_PASSWORD:-failureislame}"

APP_LABEL="app.kubernetes.io/part-of=${APP_NAME}"

init(){
# update openshift context to project
oc project ${NAMESPACE} || oc new-project ${NAMESPACE}
}

setup_db(){
# setup postgres
oc new-app \
  --name ${DB_APP_NAME} \
  -n ${NAMESPACE} \
  -l ${APP_LABEL} \
  --image-stream=postgresql:12-el8

# setup postgres env
oc set env \
  deployment/${DB_APP_NAME} \
  -n ${NAMESPACE} \
  -e POSTGRESQL_DATABASE=${DB_DATABASE} \
  -e POSTGRESQL_USER=${DB_USERNAME} \
  -e POSTGRESQL_PASSWORD=${DB_PASSWORD}

# make db persistent
oc set volume \
  deployment/${DB_APP_NAME} \
  --add \
  --name=${DB_APP_NAME} \
  --mount-path=/var/lib/postgresql/data \
  -t pvc \
  --claim-size=1G \
  --overwrite
}

setup_db_data(){
POD=$(oc get pod -l deployment="${DB_APP_NAME}" -o name | sed 's#pod/##')

echo "POD: ${POD}"
oc -n "${NAMESPACE}" cp db.sql "${POD}":/tmp
oc -n "${NAMESPACE}" cp sensor.csv.zip "${POD}":/tmp

oc -n "${NAMESPACE}" exec "${POD}" -- sh -c "$(cat -)" << COMMAND
# you can run the following w/ oc rsh
# this hack just saves you time

cd /tmp
# curl url.zip > sensor.csv.zip
unzip -o sensor.csv.zip
psql -d edge-db -f db.sql

COMMAND
}

setup_app(){
# setup prediction app
oc new-app \
  https://github.com/Enterprise-Neurosystem/edge-failure-prediction.git#${GIT_BRANCH} \
  --name ${APP_NAME} \
  -l ${APP_LABEL} \
  -n ${NAMESPACE} \
  --context-dir src

# setup database parameters
oc set env \
  deployment/${APP_NAME} \
  -e ${DB_HOSTNAME} \
  -e ${DB_DATABASE} \
  -e ${DB_USERNAME} \
  -e ${DB_PASSWORD}

# create route
oc expose service \
  ${APP_NAME} \
  -n ${NAMESPACE} \
  -l ${APP_LABEL} \
  --overrides='{"spec":{"tls":{"termination":"edge"}}}'
}

init
setup_db
# setup_db_data
setup_app
