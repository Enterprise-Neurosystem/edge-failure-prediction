# edge-db

## Command Dump

Setup local port forwarding to database (localhost:5432)

```
oc -n edge-db \
  port-forward \
  svc/edge-db 5432:5432
```

```
setup.sh
```
