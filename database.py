from process import WaveClass
import os
import psycopg2
import numpy as np
import falconn


def init_db():
    """
    Usage: create two tables or overwrite the existing ones.
    The first table is called idx_table, which contains the songs' name and their index.

    The second table is called 'songs' table,
    which contains peaks in periodogram for all windows,
    as well as the index of the song and the index of the line.
    Each record in the database corresponds to one list of peaks.

    These two tables are linked by using 'idx' of the song.
    Index of the window is not stored because we can
    query the song table and got the line index of the first match.
    Difference+1 between the two line indexes would be the index of the window.

    :return: no returns.
    """
    try:
        connect = psycopg2.connect(database="music", user="honka",
                                   password="daichan224",
                                   host="localhost")
        cursor = connect.cursor()

    except psycopg2.DatabaseError:
        if connect:
            connect.rollback()
        return False

    cursor.execute("DROP TABLE IF EXISTS idx_table")
    cursor.execute("DROP TABLE IF EXISTS songs")

    cursor.execute("""CREATE TABLE songs 
                   (line_id INTEGER PRIMARY KEY, id INTEGER, fingerprint FLOAT[]);""")

    cursor.execute("""CREATE TABLE idx_table 
                   (id INTEGER PRIMARY KEY, song_name VARCHAR);""")
    connect.commit()
    connect.close()
    return True


def build_tables(directory):
    """
    Usage: Insert periodograms, or other kinds of transformed data
    into the 'songs' table as 'fingerprint'. Windows in one song would
    share the same value of 'id', which is the index of the song.

    Create idx_table which contains the index of the song and songs' name.
    If we want to add more information into the database,
    see the function called add_more_info.

    :param directory: The path to the folder where songs are stored.

    """
    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()

    line_id = 1
    namelist = [os.path.join(directory, f) for f in os.listdir(directory)]
    for i in range(1, len(namelist)):
        cursor.execute("""INSERT INTO idx_table(id, song_name) VALUES (%s, %s)""", (i, namelist[i]))

    for i in range(1, len(namelist)):
        print(namelist[i])
        wv = WaveClass(namelist[i])
        for j in range(len(wv.trans_data)):
            list_data = wv.trans_data[j].tolist()
            cursor.execute("""INSERT INTO songs 
                           (line_id, id, fingerprint) VALUES (%s, %s, %s)""", (line_id, i, list_data))
            line_id += 1

    connect.commit()
    connect.close()
    return True


def query_song(LineID):
    """
    Usage: query the index of a song and the fingerprint
    of a window by using the index of the line.

    This function is used after hashing and matching.

    :param LineID:
    :return: tuple(index of the song, fingerprint of one window)
    """
    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()
    cursor.execute("SELECT id, fingerprint FROM songs WHERE line_id = %s", (LineID,))
    rtn = cursor.fetchone()
    connect.close()
    return rtn


def query_song_info(index):
    """
    Usage: get the information stored in idx_table by using song's index
    :param id:
    :return: information of one song
    """
    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()
    cursor.execute("SELECT * FROM idx_table WHERE id = %s", (index,))
    rtn = cursor.fetchone()
    connect.close()
    return rtn


def get_all_fingerprint():
    """
    Usage: get fingerprints stored in the database for all the windows,
    transform the data into a list with a type of np.float.
    :return: a list of all the fingerprints
    """
    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()
    # return: list of tuples
    cursor.execute("SELECT fingerprint FROM songs;")
    fingerprint_list = cursor.fetchall()
    hash_input = []
    for fingerprint in fingerprint_list:
        hash_input.append(fingerprint[0])
    hash_input = np.asarray(hash_input, dtype=np.float32)
    connect.close()
    return hash_input


def hashing(hash_input):
    """
    Usage: generate hash code for static dataset
    :param hash_input: a list that has two dimensions.
    :return: a pointer, pointing to a falconn hash table.
    """

    parameters = falconn.get_default_parameters(len(hash_input), len(hash_input[0]))
    lsh = falconn.LSHIndex(parameters)
    lsh.setup(hash_input)
    query_table = lsh.construct_query_object()
    return query_table


def query_hash_nearest(path, query_table):
    """
    Usage: return the song that has the closet relationship with the query song.

    Method: 1. transform query song's data into periodograms.
    2. For each window in the query song, find the most similar window in the database,
    compute the distance between the two window.
    3. Pick the smallest window, get the line id of the window.
    4. Query songs table to find the index of the song that contains the window.
    5  Query the information of the song stored in idx_table by using the index of the song.

    :param path: Path to the song that needs to be queried.
    :param query_table: a falconn hash table.
    :return: information of the most similar song.
    the distance between the query song and its most similar records.
    """

    wv_q = WaveClass(path)
    nearest = []
    wv_q.trans_data = np.asarray(wv_q.trans_data, dtype=np.float32)

    for query_line in wv_q.trans_data:
        nearest.append(query_table.find_k_nearest_neighbors(query_line, 1))

    distance = []
    nearest_song_id = []
    k = 0
    for i in range(len(nearest)):
        idx = nearest[i][0] + 1
        nearest_fingerprint = np.asarray(query_song(idx)[1])
        nearest_song_id.append(query_song(idx)[0])
        distance.append(np.sum((wv_q.trans_data[k] - nearest_fingerprint)**2))
        k += 1

    min_distance = np.min(distance)
    min_distance_line = np.argmin(distance)
    min_distance_id = nearest_song_id[min_distance_line]
    info = query_song_info(min_distance_id)
    return info, min_distance


def insert_new_songs(path):
    """
    :param path:
    :return:
    """
    if detect_duplicates(path):
        return True

    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()
    cursor.execute("SELECT id FROM idx_table ORDER BY id DESC;")
    id = cursor.fetchone()
    cursor.execute("SELECT line_id FROM songs ORDER BY line_id DESC")
    line_id = cursor.fetchone()

    cursor.execute("""INSERT INTO idx_table(id, song_name) VALUES (%s, %s)""", (id+1, path))

    line_id += 1
    wv = WaveClass(path)
    for i in range(len(wv.trans_data)):
        list_data = wv.trans_data[i].tolist()
        cursor.execute("""INSERT INTO songs 
                       (line_id, id, fingerprint) VALUES (%s, %s, %s)""", (line_id+i, id+1, list_data))

    connect.commit()
    connect.close()
    return True


def insert_new_info(info_list, col_name):
    """
    Usage: insert new information into the idx_table.
    :param info_list: a list which contains the information,
    like 'author', for all the songs stored in the idx_table.
    :return: True if the insertion works.
    """
    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()

    cursor.execute("ALTER TABLE idx_table ADD " + col_name + " VARCHAR;")
    for key, info in enumerate(info_list):
        key += 1
        cursor.execute("UPDATE idx_table SET " + col_name + " = (%s) WHERE id = (%s);", (info, key))
    connect.commit()
    connect.close()
    return True


def detect_duplicates(song_name):
    """
    Usage: if the query song is in the database,
    the copy in the databse should be removed before the query.
    :param song_name:
    """
    connect = psycopg2.connect(database="music", user="honka",
                               password="daichan224",
                               host="localhost")
    cursor = connect.cursor()
    cursor.execute("SELECT id FROM idx_table WHERE song_name = (%s);", (song_name,))
    id_temp = cursor.fetchone()
    cursor.close()

    if len(id_temp) != 0:
        return True

    return False

if __name__ == '__main__':
    init_db()
    build_tables("mp3/")
