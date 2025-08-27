from queue import Queue
from threading import Thread
import sys
import os
from classifier.classifier import Classifier
from db import Database
from tagging import Track

db = Database()

base_dir_name = ""

def tag_thread(q):
    while True:
        path = q.get()
        print(f"[Main] Adding {path}")
        track = Track(path)
        dbq.put(("tag",path, track))
        q.task_done()

def db_thread(q):
    db = Database()
    while True:
        type, path, data = q.get()
        match type:
            case "path":
                db.add_path(path)
                q.task_done()
            case "sonic":
                for filepath in data:
                    for type in data[filepath]:
                        db.add_classification(filepath, data[filepath][type])
                q.task_done()
            case "tag":
                db.add_track(path,data)
                q.task_done()

def path_scan(dir):
    db = Database()
    existing = set(db.get_paths('tracks','path'))
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(('.mp3', '.flac', '.ogg',)):
                filepath = os.path.join(root, file)
                if filepath not in existing:
                    dbq.put(("path",filepath,""))
                else:
                    print(f"[Main] Path exists in db: {filepath}")
    db.remove_dead_paths()

def tag_scan():
    """Scan known paths, add tags to db"""
    db = Database()
    files = set(db.get_paths('tracks','path'))
    for file in files:
        q.put(file)

def sonic_thread(sq, classifier):
    while True:
        paths = sq.get()
        print(f"[Main] Adding {paths}")
        track_data = classifier.process_tracks(paths)
        dbq.put(("sonic","",track_data))
        sq.task_done()

def sonic_scan():
    db = Database()
    paths = db.get_paths('tracks','path')
    existing = set(db.get_paths("track_classifications","track_path"))
    batch = []
    for path in paths:
        if path not in existing:
            batch.append(path)
            if len(batch) >= 20:
                sq.put(batch)
                batch.clear()
        else:
            print(f"[Main] Path exists in db: {path}")


def make_playlist(path, percent):
    db = Database()
    top = db.get_track_classifications(path, percent)
    moods = db.get_track_mood_classification(path)
    track_info = Track(path)
    track_tags = track_info.get_tags()
    artist = ""
    title = ""
    if 'artist' in track_tags:
        artist = track_tags['artist'][0]
    if 'title' in track_tags:
        title = track_tags['title'][0]
    seen = set()
    seen_multiple = [path]
    for name,value in top:
        vals = db.get_tracks_by_classification(name,value,percent)
        for val in vals:
            if val in seen:
                seen_multiple.append(val)
            else:
                seen.add(val)
    for name in moods:
        vals = db.get_tracks_by_mood_classification(name, not moods[name])
        for val in vals:
            if val in seen_multiple:
                seen_multiple.remove(val)
    create_m3u_playlist(f"Tracks Like {artist} - {title}", seen_multiple)

def create_m3u_playlist(name, paths):
    with open(f"{name}.m3u", 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(f"{path}\n")


classifier = Classifier()

dbq = Queue(maxsize=1)
sq = Queue(maxsize=5)
q = Queue(maxsize=14)

for i in range(14):
    worker = Thread(target=tag_thread, args=(q,), daemon=True).start()
for i in range(5):
    worker = Thread(target=sonic_thread, args=(sq,classifier,), daemon=True).start()
for i in range(1):
    worker = Thread(target=db_thread, args=(dbq,), daemon=True).start()

for i, arg in enumerate(sys.argv):
    run_base = False
    run_tags = False
    run_sonic = False
    match arg:
        case "-b" | "--base":
            if i+1 > len(sys.argv):
                break
            val = sys.argv[i+1]
            base_dir_name = val
        case "-s" | "--scan":
            run_base = True
        case "-t" | "--tags":
            run_tags = True
        case "-S" | "--sonic":
            run_sonic = True
        case "-pl" | "--playlist":
            if i+1 > len(sys.argv):
                break
            val = sys.argv[i+1]
            percent = 0.1
            if i+2 < len(sys.argv) and sys.argv[i+2].startswith("0."):
                percent = float(sys.argv[i+2])
            make_playlist(val, percent)
    if run_base:
        path_scan(base_dir_name)
    if run_tags:
        tag_scan()
    if run_sonic:
        sonic_scan()

dbq.join()
sq.join()
q.join()
