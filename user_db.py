import aiosqlite
# from functools import wraps
# from aiosqlite import connect
# from bot_config import *

# from collections.abc import Sequence

# def connect_db(db_url):
#     def decorator(func):
#         wraps(func)
#
#         async def wrapper(*args, **kwargs):
#             async with connect(db_url) as conn:
#                 cursor = await conn.cursor()
#                 return await func(conn, cursor, *args, **kwargs)
#
#         return wrapper
#
#     return decorator


async def create_connection(db_dir: str):
    connection = await aiosqlite.connect(db_dir)
    return connection


async def create_table(connection):
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS "user_data" (
    "id"    INTEGER,
    "user_id"   INTEGER UNIQUE,
    "user_name" TEXT,
    "photo_1"   BLOB,
    "photo_2"   BLOB,
    "final_photo"   BLOB,
    "q_num" INTEGER DEFAULT 0,
    "r_q_num"   INTEGER DEFAULT 0,
    PRIMARY KEY("id" AUTOINCREMENT)
    );
        '''
    await connection.execute(create_table_query)


async def insert_data(connection, user_id: int, col_names: tuple[str, ...], cols_data: list[...]):
    assert len(col_names) == len(cols_data)

    # FIXME: doesn't work for one column (use update_cols instead)
    insert_query = f"""
        INSERT INTO user_data {col_names}
        SELECT {', '.join('?' * len(col_names))}
        WHERE NOT EXISTS (SELECT 2 FROM user_data WHERE user_id = ?);"""

    await connection.execute(insert_query, (*cols_data, user_id))
    await connection.commit()


async def execute_query(connection, query: str, vals: tuple = tuple(), q_type: str = 'one'):
    async with connection.execute(query, vals) as cursor:
        result = (await cursor.fetchone())[0] if q_type == 'one' else await cursor.fetchall()
    return result


async def get_all_data(connection):
    select_query = 'SELECT * FROM user_data'
    result = await execute_query(connection, select_query)
    return result


async def update_cols(connection, user_id, col_names: list[str, ...], cols_data: list):
    assert len(col_names) == len(cols_data)

    update_query = f"""UPDATE user_data SET {", ".join([f'{col} = ?' for col in col_names])} WHERE user_id = ?"""
    await connection.execute(update_query, (*cols_data, user_id))
    await connection.commit()


async def get_bphotos(conn, user_id):
    q_text_1 = f'SELECT photo_1 FROM user_data WHERE user_id = ?'
    q_text_2 = f'SELECT photo_2 FROM user_data WHERE user_id = ?'

    bphoto_1 = await execute_query(conn, q_text_1, (user_id,))
    bphoto_2 = await execute_query(conn, q_text_2, (user_id,))

    return bphoto_1, bphoto_2


async def delete_user(connection, user_id):
    delete_query = 'DELETE FROM user_data WHERE id = ?'
    await connection.execute(delete_query, (user_id,))
    await connection.commit()
