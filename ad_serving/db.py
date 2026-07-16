"""MySQL database wrapper for the ad serving system."""

import os
from contextlib import contextmanager

import pymysql
import pymysql.cursors


class Database:
    def __init__(self):
        self.config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 3306)),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "ad_serving"),
            "cursorclass": pymysql.cursors.DictCursor,
            "autocommit": True,
        }

    @contextmanager
    def connection(self):
        conn = pymysql.connect(**self.config)
        try:
            yield conn
        finally:
            conn.close()

    def fetchone(self, query: str, args=None):
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, args)
                return cur.fetchone()

    def fetchall(self, query: str, args=None):
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, args)
                return cur.fetchall()

    def execute(self, query: str, args=None):
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, args)
                return cur.lastrowid

    def executemany(self, query: str, args_list):
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, args_list)
                return cur.lastrowid
