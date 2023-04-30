#!/bin/sh

# setup app parameters
APP_NAME="${APP_NAME:-predict}"
NAMESPACE="${NAMESPACE:-edge-failure-prediction}"
GIT_BRANCH="workshop/updates"

# setup database parameters
DB_APP_NAME="${DB_APP_NAME:-predict-db}"
DB_HOSTNAME="${DB_HOSTNAME:-${DB_APP_NAME}.${NAMESPACE}.svc.cluster.local}"
DB_DATABASE="${DB_DATABASE:-predict-db}"
DB_USERNAME="${DB_USERNAME:-predict-db}"
DB_PASSWORD="${DB_PASSWORD:-failureislame}"
DB_PATH=database

APP_LABEL="app.kubernetes.io/part-of=${APP_NAME}"

init(){
# update openshift context to project
oc project ${NAMESPACE} || oc new-project ${NAMESPACE}
}

setup_db_instance(){
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
  --claim-name=${DB_APP_NAME} \
  --overwrite
}

setup_db_data(){
POD=$(oc get pod -l deployment="${DB_APP_NAME}" -o name | sed 's#pod/##')

echo "copying data to database container..."
echo "POD: ${POD}"
# oc -n "${NAMESPACE}" cp "${DB_PATH}"/db.sql "${POD}":/tmp
# oc -n "${NAMESPACE}" cp "${DB_PATH}"/sensor.csv.zip "${POD}":/tmp

cat << COMMAND | oc -n "${NAMESPACE}" exec "${POD}" -- sh -c "$(cat -)"
# you can run the following w/ oc rsh
# this hack just saves you time

cd /tmp
# curl url.zip > sensor.csv.zip
unzip -o sensor.csv.zip

echo 'GRANT ALL ON TABLE waterpump TO "'"${DB_USERNAME}"'" ;' >> db.sql
psql -d $DB_DATABASE -f db.sql

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
  -e DB_HOSTNAME=${DB_HOSTNAME} \
  -e DB_DATABASE=${DB_DATABASE} \
  -e DB_USERNAME=${DB_USERNAME} \
  -e DB_PASSWORD=${DB_PASSWORD}

# create route
oc expose service \
  ${APP_NAME} \
  -n ${NAMESPACE} \
  -l ${APP_LABEL} \
  --overrides='{"spec":{"tls":{"termination":"edge"}}}'

# kludge - fix timeout for app
oc annotate route \
  ${APP_NAME} \
  -n ${NAMESPACE} \
  haproxy.router.openshift.io/timeout=5m \
  --overwrite
}

setup_db(){
  echo "If you are participating in a workshop answer NO."
  read -r -p "Setup sensor data database? [y/N] " input
  case $input in
    [yY][eE][sS]|[yY])
      setup_db
      setup_db_data
      ;;
    [nN][oO]|[nN])
      echo
      ;;
    *)
      echo
      ;;
  esac
  setup_db_instance
  setup_db_data
}

init
setup_db
setup_app
