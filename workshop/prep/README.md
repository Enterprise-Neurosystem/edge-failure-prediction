# edge-db

## command dump

```
oc -n edge-db \
  port-forward \
  svc/edge-db 5432:5432
```

```
oc cp ../dump/waterpumpschema.sql edge-db-0:/tmp
oc cp sensor.csv.zip edge-db-0:/tmp
oc rsh edge-db-0
```

```
cd /tmp
unzip sensor.csv.zip
psql -d edge-db -f waterpumpschema.sql
```
