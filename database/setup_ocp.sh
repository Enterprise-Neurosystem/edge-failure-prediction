#!/bin/sh
# set -x

APP_NAME=${APP_NAME:-edge-db}
NAMESPACE=${NAMESPACE:-edge-db}

ocp_upload_files(){
  POD=$(oc get pod -l deployment="${APP_NAME}" -o name | sed 's#pod/##')

  echo "POD: ${POD}"
  oc -n "${NAMESPACE}" cp db.sql "${POD}":/tmp
  oc -n "${NAMESPACE}" cp sensor.csv.zip "${POD}":/tmp
}

ocp_setup_db(){
cat << EOL | oc -n "${NAMESPACE}" exec "${POD}" -- sh -c "$(cat -)"

# you can run the following w/ oc rsh
# this hack just saves you time

cd /tmp
# curl url.zip > sensor.csv.zip
unzip -o sensor.csv.zip
psql -d edge-db -f db.sql

EOL
}

ocp_upload_files
ocp_setup_db
