-- https://stackoverflow.com/questions/2987433/how-to-import-csv-file-data-into-a-postgresql-table
DROP TABLE IF EXISTS waterpump;
CREATE TABLE waterpump 
    (id serial NOT NULL PRIMARY KEY,
    timestamp TIMESTAMP, 
    sensor_00 DOUBLE PRECISION,
    sensor_01 DOUBLE PRECISION,
    sensor_02 DOUBLE PRECISION,
    sensor_03 DOUBLE PRECISION,
    sensor_04 DOUBLE PRECISION,
    sensor_05 DOUBLE PRECISION,
    sensor_06 DOUBLE PRECISION,
    sensor_07 DOUBLE PRECISION,
    sensor_08 DOUBLE PRECISION,
    sensor_09 DOUBLE PRECISION,
    sensor_10 DOUBLE PRECISION,
    sensor_11 DOUBLE PRECISION,
    sensor_12 DOUBLE PRECISION,
    sensor_13 DOUBLE PRECISION,
    sensor_14 DOUBLE PRECISION,
    sensor_15 DOUBLE PRECISION,
    sensor_16 DOUBLE PRECISION,
    sensor_17 DOUBLE PRECISION,
    sensor_18 DOUBLE PRECISION,
    sensor_19 DOUBLE PRECISION,
    sensor_20 DOUBLE PRECISION,
    sensor_21 DOUBLE PRECISION,
    sensor_22 DOUBLE PRECISION,
    sensor_23 DOUBLE PRECISION,
    sensor_24 DOUBLE PRECISION,
    sensor_25 DOUBLE PRECISION,
    sensor_26 DOUBLE PRECISION,
    sensor_27 DOUBLE PRECISION,
    sensor_28 DOUBLE PRECISION,
    sensor_29 DOUBLE PRECISION,
    sensor_30 DOUBLE PRECISION,
    sensor_31 DOUBLE PRECISION,
    sensor_32 DOUBLE PRECISION,
    sensor_33 DOUBLE PRECISION,
    sensor_34 DOUBLE PRECISION,
    sensor_35 DOUBLE PRECISION,
    sensor_36 DOUBLE PRECISION,
    sensor_37 DOUBLE PRECISION,
    sensor_38 DOUBLE PRECISION,
    sensor_39 DOUBLE PRECISION,
    sensor_40 DOUBLE PRECISION,
    sensor_41 DOUBLE PRECISION,
    sensor_42 DOUBLE PRECISION,
    sensor_43 DOUBLE PRECISION,
    sensor_44 DOUBLE PRECISION,
    sensor_45 DOUBLE PRECISION,
    sensor_46 DOUBLE PRECISION,
    sensor_47 DOUBLE PRECISION,
    sensor_48 DOUBLE PRECISION,
    sensor_49 DOUBLE PRECISION,
    sensor_50 DOUBLE PRECISION,
    sensor_51 DOUBLE PRECISION,
    machine_status TEXT);

GRANT ALL ON TABLE public.waterpump TO "edge-db";

COPY waterpump
    FROM '/tmp/sensor.csv'
    WITH CSV HEADER DELIMITER ',' ;

ALTER TABLE waterpump
    DROP COLUMN id ;
