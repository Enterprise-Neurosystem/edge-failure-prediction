# edge-db

## Run postgres as a container locally w/ podman (docker)

Use the following for local development.

Note: The container will automatically be removed when it is stopped

```
./setup.sh
```

```
docker rm edge-db
```

## Run postgres in OpenShift

Setup new project for postgres DB in OpenShift

Note: Modify the (export) env vars below to fit your needs

```
# setup parameters
export APP_NAME=edge-db
export NAMESPACE=edge-failure-prediction
export SVC_NAME="${APP_NAME}.${NAMESPACE}.svc.cluster.local"

APP_LABEL="app.kubernetes.io/part-of=${APP_NAME}"
```

```
# update openshift context to project
oc project ${NAMESPACE} || oc new-project ${NAMESPACE}
```

```
# setup postgres
oc new-app \
  --name ${APP_NAME} \
  -n ${NAMESPACE} \
  -l ${APP_LABEL} \
  --image-stream=postgresql:12-el8

# setup postgres env
oc set env \
  deployment/${APP_NAME} \
  -n ${NAMESPACE} \
  -e POSTGRESQL_DATABASE=edge-db \
  -e POSTGRESQL_PASSWORD=failure \
  -e POSTGRESQL_USER=edge-db

# make db persistent
oc set volume \
  deployment/${APP_NAME} \
  --add \
  --name=${APP_NAME} \
  --mount-path=/var/lib/postgresql/data \
  -t pvc \
  --claim-size=1G \
  --overwrite
```

Use [setup.sh](setup.sh) to setup data in DB

```
./setup_ocp.sh
```

Setup local port forwarding to database (localhost:5432) on OpenShift

You can connect to your database in OpenShift via `localhost:5432` while this command is running

```
oc -n ${NAMESPACE} \
  port-forward \
  svc/${APP_NAME} 5432:5432
```

Run commands in postgres container

```
oc rsh deployment/${APP_NAME}

# ex: interact with cli
# psql -d edge-db
# SELECT * FROM waterpump ORDER BY timestamp;
```
