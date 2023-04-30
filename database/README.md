# Database Info

## Run postgres as a container locally w/ podman (docker)

Use the following to start a container for local development.

Note: The container will automatically be removed when it is stopped

```
# NOTICE: run from repo root dir

. scripts/bootstrap.sh
container_setup_db
```

## Run postgres in OpenShift

```
# NOTICE: run from repo root dir

. scripts/bootstrap.sh
ocp_setup_db
```

You can connect to your OpenShift database via `localhost:5432`
while the following `oc port-forward` command is running.
Update the parameters as appropriate

```
DB_APP_NAME="predict-db"
NAMESPACE="$(oc project)"

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
