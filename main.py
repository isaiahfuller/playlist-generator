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
    while True:
        task = q.get()
        try:
            db = Database()
            if task[0] == "tag":
                _type, path, track_data = task
                db.add_track(path, track_data) # Assuming add_track is the correct method based on original code
            elif task[0] == "path":
                _type, path, _data = task
                db.add_path(path)
            elif task[0] == "sonic":
                _type, _path, data = task # The path is an empty string here, data contains the dict
                for filepath in data:
                    for classification_type in data[filepath]:
                        db.add_classification(filepath, data[filepath][classification_type])
            db.close()
        except Exception as e:
            print(f"[Database] Thread error: {e}", flush=True)
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
    existing = set(db.get_paths('track_tags','track_path'))
    for file in files:
        if file not in existing:
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
    db.close()
    
    batch = []
    for path in paths:
        if path not in existing:
            batch.append(path)
            if len(batch) >= 1:
                sq.put(batch)
                batch = []
        # else:
        #     print(f"[Main] Path exists in db: {path}")
    print("[Main] sonic_scan finished. Waiting for threads...", flush=True)


def get_neighbors(db, path, percent=0.1, limit=20):
    top = db.get_track_classifications(path, percent)
    moods = db.get_track_mood_classification(path)
    
    seen = set()
    seen_multiple = set([path])
    
    for name,value in top:
        vals = db.get_tracks_by_classification(name,value,0.1)
        for val in vals:
            if val in seen:
                seen_multiple.add(val)
            else:
                seen.add(val)
    
    for name in moods:
        vals = db.get_tracks_by_mood_classification(name, not moods[name])
        for val in vals:
            if val in seen_multiple:
                seen_multiple.remove(val)
                
    if path in seen_multiple:
        seen_multiple.remove(path)
        
    # Limit the number of neighbors to avoid explosion
    neighbors = list(seen_multiple)
    if len(neighbors) > limit:
        return neighbors[:limit]
    return neighbors

def get_similar_songs(path, percent=0.1, min_tracks=2):
    db = Database()
    track_info = Track(path)
    track_tags = track_info.get_tags()
    artist = ""
    title = ""
    if 'artist' in track_tags:
        artist = track_tags['artist'][0]
    if 'title' in track_tags:
        title = track_tags['title'][0]
        
    neighbors = get_neighbors(db, path, percent, limit=100) # Higher limit for single playlist generation
    paths = [path] + neighbors
    
    if len(paths) < min_tracks:
        print("[Main] Not enough tracks found for playlist, rerunning with lower percentage", flush=True)
        get_similar_songs(path, percent-0.01)
        return
    create_m3u_playlist(f"Tracks Like {artist} - {title}", paths)

def find_path(db, start_path, end_path, min_tracks=5, percent=0.08, limit=20):
    # Bidirectional BFS
    queue_start = [start_path]
    visited_start = {start_path: [start_path]}
    
    queue_end = [end_path]
    visited_end = {end_path: [end_path]}
    
    while queue_start and queue_end:
        # Expand start side
        if queue_start:
            current_start = queue_start.pop(0)
            path_start = visited_start[current_start]
            
            # Check if connected
            if current_start in visited_end:
                path_end = visited_end[current_start]
                full_path = path_start + path_end[::-1][1:]
                if len(full_path) >= min_tracks:
                    return full_path
            
            neighbors = get_neighbors(db, current_start, percent, limit=limit)
            for neighbor in neighbors:
                if neighbor not in visited_start:
                    visited_start[neighbor] = path_start + [neighbor]
                    queue_start.append(neighbor)

        # Expand end side
        if queue_end:
            current_end = queue_end.pop(0)
            path_end = visited_end[current_end]
            
            # Check if connected
            if current_end in visited_start:
                path_start = visited_start[current_end]
                full_path = path_start + path_end[::-1][1:]
                if len(full_path) >= min_tracks:
                    return full_path
                
            neighbors = get_neighbors(db, current_end, percent, limit=limit)
            for neighbor in neighbors:
                if neighbor not in visited_end:
                    visited_end[neighbor] = path_end + [neighbor]
                    queue_end.append(neighbor)
    return None

def generate_transition_playlist(tracks, min_tracks=5):
    if len(tracks) < 2:
        print("[Main] Need at least 2 tracks for a transition.", flush=True)
        return

    print(f"[Main] Generating transition playlist for {len(tracks)} tracks...", flush=True)
    db = Database()
    
    full_path = []
    
    for i in range(len(tracks) - 1):
        start = tracks[i]
        end = tracks[i+1]
        print(f"[Main] Finding path from {start} to {end}...", flush=True)
        
        segment_min = max(2, min_tracks // (len(tracks) - 1))
        
        # Retry logic
        # (percent, limit)
        retry_configs = [
            (0.1, 100),  # Default
            (0.08, 200),   # Relaxed
            (0.05, 500),   # Very Relaxed
            (0.03, 2000),   # Last Resort
        ]
        
        path = None
        for attempt, (p, l) in enumerate(retry_configs):
            print(f"[Main] Attempt {attempt+1}/{len(retry_configs)}: Finding path from {start} to {end} (percent={p}, limit={l})...", flush=True)
            path = find_path(db, start, end, min_tracks=segment_min, percent=p, limit=l)
            if path:
                break
            else:
                print(f"[Main] Attempt {attempt+1} failed.", flush=True)
        
        if path:
            print(f"[Main] Segment found with {len(path)} tracks.", flush=True)
            if full_path:
                # Avoid duplicating the connection node
                full_path.extend(path[1:])
            else:
                full_path.extend(path)
        else:
            print(f"[Main] No path found between {start} and {end} after all retries. Aborting.", flush=True)
            return

    if full_path:
        print(f"[Main] Full path found with {len(full_path)} tracks.", flush=True)
        # Get names for filename
        start_track = Track(tracks[0])
        end_track = Track(tracks[-1])
        start_tags = start_track.get_tags()
        end_tags = end_track.get_tags()
        
        start_name = start_tags.get('title', ['Start'])[0]
        end_name = end_tags.get('title', ['End'])[0]
        
        create_m3u_playlist(f"Transition {start_name} to {end_name}", full_path)
    else:
        print("[Main] Failed to generate playlist.", flush=True)

def create_m3u_playlist(name, paths):
    with open(f"{name.replace('/', '_')}.m3u", 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(f"{path}\n")


if __name__ == "__main__":
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
                run_base = True
            case "-t" | "--tags":
                run_tags = True
            case "-s" | "--sonic":
                run_sonic = True
            case "-pl" | "--playlist":
                if i+1 > len(sys.argv):
                    break
                val = sys.argv[i+1]
                percent = 0.1
                if i+2 < len(sys.argv) and sys.argv[i+2].startswith("0."):
                    percent = float(sys.argv[i+2])
                get_similar_songs(val, percent)
            case "-tr" | "--transition":
                if i+2 > len(sys.argv):
                    print("Usage: --transition <track1> <track2> [track3 ...]")
                    break
                tracks = []
                length = 5
                for j in range(i+1, len(sys.argv)):
                    arg = sys.argv[j]
                    if arg.startswith("-"):
                        break
                    tracks.append(arg)
                
                # Check for length flag in the rest of arguments
                if i + 1 + len(tracks) < len(sys.argv):
                     next_arg_idx = i + 1 + len(tracks)
                     if sys.argv[next_arg_idx] in ["-l", "--length"]:
                         if next_arg_idx + 1 < len(sys.argv):
                             length = int(sys.argv[next_arg_idx+1])

                if len(tracks) < 2:
                     print("Usage: --transition <track1> <track2> [track3 ...]")
                     break
                     
                generate_transition_playlist(tracks, min_tracks=length)
        if run_base:
            path_scan(base_dir_name)
        if run_tags:
            tag_scan()
        if run_sonic:
            sonic_scan()

    q.join()
    sq.join()
    dbq.join()
