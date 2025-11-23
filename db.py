import sqlite3
import os
import glob
import json
import itertools
import base64

from tagging import Track

class Database:
    def __init__(self):
        self._classes = {}
        db_exists = os.path.exists("test.db")
        self.__con = sqlite3.connect("test.db", timeout=10.0, isolation_level=None)
        self.__con.execute("PRAGMA journal_mode=WAL")
        if not db_exists:
            self.create_tables()
            self.create_indexes()
            self.get_classifier_classes()

    def add_classification(self,path,data):
        print(f"[Database] Adding {path} ...", flush=True)
        
        # Ensure track exists in tracks table
        try:
            self.__con.execute("INSERT OR IGNORE INTO tracks (path) VALUES (?)", (path,))
        except Exception as e:
            print(f"[Database] Failed to insert track path: {e}", flush=True)

        cur = self.__con.cursor()
        success_count = 0
        error_count = 0
        for key in data:
            try:
                cur.execute('INSERT OR REPLACE INTO track_classifications VALUES (?,?,?)', (key, path, data[key]))
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"[Database] Error inserting classification {key} for {path}: {e}", flush=True)
        
        try:
            # self.__con.commit()
            pass
        except Exception as e:
            print(f"[Database] COMMIT FAILED: {e}", flush=True)

    def close(self):
        self.__con.close()

    def create_tables(self):
        # Check if the table already exists
        cur = self.__con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
        if cur.fetchone():
            print("[Database] Table 'tracks' already exists")
        else:
            cur.execute('CREATE TABLE "tracks" ("id" INTEGER, "path" text NOT NULL UNIQUE, PRIMARY KEY("id" AUTOINCREMENT) ON CONFLICT REPLACE)')
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='classifications'")
        if cur.fetchone():
            print("[Database] Table 'classifications' already exists")
        else:
            cur.execute('CREATE TABLE "classifications" ("id" INTEGER, "name" TEXT NOT NULL UNIQUE,PRIMARY KEY("id" AUTOINCREMENT))')
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='track_classifications'")
        if cur.fetchone():
            print("[Database] Table 'track_classifications' already exists")
        else:
            cur.execute('CREATE TABLE "track_classifications" ("classification_name" TEXT NOT NULL,"track_path" TEXT NOT NULL,"value" REAL NOT NULL,PRIMARY KEY("classification_name","track_path") ON CONFLICT REPLACE,FOREIGN KEY("classification_name") REFERENCES "classifications"("name"),FOREIGN KEY("track_path") REFERENCES "tracks"("path"))')
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'")
        if cur.fetchone():
            print("[Database] Table 'tags' already exists")
        else:
            cur.execute('CREATE TABLE "tags" ("id" INTEGER NOT NULL UNIQUE,"name" TEXT NOT NULL UNIQUE,PRIMARY KEY("id" AUTOINCREMENT) ON CONFLICT IGNORE)')
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='track_tags'")
        if cur.fetchone():
            print("[Database] Table 'track_tags' already exists")
        else:
            cur.execute('CREATE TABLE "track_tags" ("track_path" TEXT NOT NULL,"tag_id" INTEGER NOT NULL,"value" TEXT,FOREIGN KEY("tag_id") REFERENCES "tags"("id"),FOREIGN KEY("track_path") REFERENCES "tracks"("path"))')
        # self.__con.commit()

    def create_indexes(self):
        cur = self.__con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='tracks' AND name='track_path'")
        if cur.fetchone():
            print("[Database] Index 'track_path' on 'tracks' already exists")
        else:
            cur.execute('CREATE INDEX "track_path" ON "tracks" ("path")')
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='track_classifications' AND name='classification_name_value'")
        if cur.fetchone():
            print("[Database] Index 'classification_name_value' on 'track_classifications' already exists")
        else:
            cur.execute('CREATE INDEX "classification_name_value" ON "track_classifications" ("classification_name","value")')
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='track_classifications' AND name='classification_name_values_desc'")
        if cur.fetchone():
            print("[Database] Index 'classification_name_values_desc' on 'track_classifications' already exists")
        else:
            cur.execute('CREATE INDEX "classification_name_values_desc" ON "track_classifications" ("value" DESC, "track_path")')
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='track_classifications' AND name='classification_path'")
        if cur.fetchone():
            print("[Database] Index 'classification_path' on 'track_classifications' already exists")
        else:
            cur.execute('CREATE INDEX "classification_path" ON "track_classifications" ("track_path")')
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='track_tags' AND name='tag_path_value'")
        if cur.fetchone():
            print("[Database] Index 'tag_path_value' on 'track_tags' already exists")
        else:
            cur.execute('CREATE INDEX "tag_path_value" ON "track_tags" ("value", "track_path")')
        # self.__con.commit()

    def add_tracks(self, paths):
        if paths is None:
            raise ValueError("paths cannot be None")
        elif not isinstance(paths, list):
            raise TypeError("paths must be a list")
        elif len(paths) == 0:
            print("[Database] No tracks to add")

        cur = self.__con.cursor()
        for path in paths:
            checkPath = cur.execute("SELECT path FROM tracks WHERE path=?", (path,))
            if checkPath.fetchone():
                print(f"[Database] File at {path} already present.")
            else:
                cur.execute('INSERT INTO track (path) VALUES (?)', (path,))
                print(f'[Database] Added track "{path}" to db')
        # self.__con.commit()

    def get_classifier_classes(self):
        classifications = []
        for filepath in glob.iglob("./classifier/**/*.json", recursive=True):
            with open(filepath, "r") as f:
                data = json.load(f)
                classifications.append(data["classes"])
        flattened = list(itertools.chain.from_iterable(classifications))
        classes_ids = self.add_and_return('classifications', flattened)
        self.classifications = {}
        for id,name in classes_ids:
            self.classifications[name] = id
        return self.classifications

    def add_and_return(self, table_name, values):
        cur = self.__con.cursor()
        for value in values:
            try:
                cur.execute(f'INSERT OR IGNORE INTO {table_name} (name) VALUES (?)', (value,))
            except:
                print(f'[Database] Error on "{value}"')
        # self.__con.commit()
        cur.execute(f"SELECT * FROM {table_name}")
        res = cur.fetchall()
        # self.__con.commit() # Release read lock
        return res

    def get_tags(self, tag_names):
        """get tags from db, add new ones, get tag ids"""
        tags = self.add_and_return('tags', tag_names)
        self.tags = {}
        for id, name in tags:
            self.tags[name] = id
        return self.tags

    def add_path(self,path):
        cur = self.__con.cursor()
        cur.execute('INSERT OR IGNORE INTO tracks (path) VALUES (?)', (path,))
        # self.__con.commit()
        print(f"[Database] Added {path}")
        return

    def add_track(self, path, track):
        cur = self.__con.cursor()
        track_tags = track.get_tags()
        if track_tags == None:
            return None
        self.get_tags(list(track_tags.keys()))
        cur.execute('INSERT OR IGNORE INTO tracks (path) VALUES (?)', (path,))
        cur.execute('DELETE FROM track_tags WHERE track_path=?', (path,))
        for tag_name in track_tags:
            for value in track_tags[tag_name]:
                encoded = value.encode("utf-8")
                try:
                    cur.execute('INSERT OR IGNORE INTO track_tags VALUES (?,?,?)', (path, tag_name, base64.b64encode(encoded)))
                except:
                    print(f"[Database] Error adding {tag_name}: {value} on {path}")
        # self.__con.commit()
        print(f"[Database] Added {path}")

    def remove_dead_paths(self):
        cur = self.__con.cursor()
        tracks = self.get_paths('tracks','path')
        for track in tracks:
            if not os.path.exists(track):
                cur.execute('DELETE FROM track_tags WHERE track_path = ?', (track,))
                cur.execute('DELETE FROM track_classifications WHERE track_path = ?', (track,))
                cur.execute('DELETE FROM tracks WHERE path = ?', (track,))
        # self.__con.commit()

    def add_classification(self,path,data):
        cur = self.__con.cursor()
        success_count = 0
        error_count = 0
        for i, key in enumerate(data):
            try:
                cur.execute('INSERT OR REPLACE INTO track_classifications VALUES (?,?,?)', (key, path, data[key]))
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"[Database] Error inserting classification {key} for {path}: {e}", flush=True)
        
        # try:
        #     self.__con.commit()
        #     print(f'[Database] Commit successful. Added "{path}" classification to db', flush=True)
        # except Exception as e:
        #     print(f"[Database] COMMIT FAILED: {e}", flush=True)

    def get_paths(self, table_name, col_name) -> list[str]:
        cur = self.__con.cursor()
        cur.row_factory = lambda _c, r: r[0]
        cur.execute(f"SELECT {col_name} FROM {table_name}")
        res = cur.fetchall()
        return res

    def get_track_classifications(self, path, percent):
        cur = self.__con.cursor()
        cur.execute('SELECT classification_name,value FROM track_classifications WHERE track_path = ? AND value >= ? AND classification_name LIKE "%---%"', (path, percent))
        res = cur.fetchall()
        return res

    def get_track_mood_classification(self,path):
        cur = self.__con.cursor()
        cur.execute('select classification_name,value from track_classifications where track_path = ? AND classification_name NOT LIKE "%---%" AND classification_name NOT LIKE "non_%"', (path,))
        vals = cur.fetchall()
        res = {}
        for name,val in vals:
            res[name] = val > 0.5
        return res

    def get_tracks_by_classification(self, name, value, percent):
        cur = self.__con.cursor()
        cur.row_factory = lambda _c, r: r[0]
        cur.execute('SELECT DISTINCT track_path FROM track_classifications WHERE classification_name = ? AND value >= ? AND value <= ?', (name, value - (value*percent), value + (value*percent)))
        res = cur.fetchall()
        return res

    def get_tracks_by_mood_classification(self, name, value):
        cur = self.__con.cursor()
        cur.row_factory = lambda _c, r: r[0]
        operator = ">=" if value else "<="
        cur.execute(f'SELECT DISTINCT track_path FROM track_classifications WHERE classification_name = ? AND value {operator} 0.5', (name,))
        res = cur.fetchall()
        return res
