from services.connection_pool_singleton import ConnectionPoolSingleton
import pandas as pd
import psycopg2


class KaggleDataService:
    GET_ALL_ROWS = "Select * from waterpump order by timestamp"

    @staticmethod
    def get_all_as_df():
        pool = ConnectionPoolSingleton.getConnectionPool()
        connection = pool.getconn()
        try:
            with connection:
                df = pd.read_sql_query(KaggleDataService.GET_ALL_ROWS, connection)
                df.set_index("timestamp", inplace=True)
        except (Exception, psycopg2.DatabaseError) as err:
            print(err)
        else:
            return df
        finally:
            pool.putconn(connection)
