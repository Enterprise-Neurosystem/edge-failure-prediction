# edge-db

## Run postgres as a container locally w/ podman (docker)

Use the following to start a container for local development.

Note: The container will automatically be removed when it is stopped

```
# run from repo root
# cd ..

. scripts/bootstrap.sh
podman_setup_db
```

## Run postgres in OpenShift

```
# run from repo root
# cd ..

. scripts/bootstrap.sh
ocp_setup_db
```

Setup local port forwarding to database (localhost:5432) on OpenShift

You can connect to your database in OpenShift via `localhost:5432` while this command is running

```
oc -n ${NAMESPACE} \
  port-forward \
  svc/${DB_APP_NAME} 5432:5432
```

Run commands in postgres container

```
oc rsh deployment/${DB_APP_NAME}

# ex: interact with cli
# psql -d predict-db
# SELECT * FROM waterpump ORDER BY timestamp;
```
