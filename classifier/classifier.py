import numpy as np
import json
from essentia.standard import (
    MonoLoader,  # pyright: ignore[reportAttributeAccessIssue]
    TensorflowPredictEffnetDiscogs,  # pyright: ignore[reportAttributeAccessIssue]
    TensorflowPredict2D,  # pyright: ignore[reportAttributeAccessIssue]
)
import shutil
import tempfile
import os
import threading

# Global lock for directory changes
dir_lock = threading.Lock()

class Classifier:
    # Our models take audio streams at 16kHz
    sr = 16000
    rq = 4

    def __init__(self):
        """Initialize classifier with Essentia models"""
        print("[Classifier] Loading primary embedding models with Essentia")
        self.discogs_model = TensorflowPredictEffnetDiscogs(
            graphFilename="./classifier/models/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1",
        )

        print("[Classifier] Loading downstream models with Essentia")
        
        # Use Essentia for all models
        self.genre_model = TensorflowPredict2D(
            graphFilename="./classifier/models/genre_discogs400-discogs-effnet-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        self.party_model = TensorflowPredict2D(
            graphFilename="./classifier/models/mood_party-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        self.relaxed_model = TensorflowPredict2D(
            graphFilename="./classifier/models/mood_relaxed-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        self.dance_model = TensorflowPredict2D(
            graphFilename="./classifier/models/danceability-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        self.moodtheme_model = TensorflowPredict2D(
            graphFilename="./classifier/models/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
        )

        # Cache the model labels
        self.labels_cache = {}
        # Preload all labels
        self._preload_labels()

    def _preload_labels(self):
        """Preload all label files to avoid file I/O during processing"""
        model_paths = [
            "./classifier/models/genre_discogs400-discogs-effnet-1",
            "./classifier/models/mood_party-discogs-effnet-1",
            "./classifier/models/mood_relaxed-discogs-effnet-1",
            "./classifier/models/danceability-discogs-effnet-1",
            "./classifier/models/mtg_jamendo_moodtheme-discogs-effnet-1",
        ]
        for path in model_paths:
            self._get_labels(path)
        print(f"[Classifier] Preloaded {len(self.labels_cache)} label sets")

    def _get_labels(self, model_name):
        """Get labels from cache or load from file if not cached"""
        if model_name not in self.labels_cache:
            with open("{}.json".format(model_name), "r") as file:
                data = json.load(file)
            self.labels_cache[model_name] = data["classes"]
        return self.labels_cache[model_name]

    def process_tracks(self, files: list[str]):
        """Process multiple audio files"""
        results = {}

        if not files:
            return results

        # Use larger batch size for better utilization
        batch_size = 20
        print(
            f"[Classifier] Processing {len(files)} files with batch size {batch_size}"
        )

        # Phase 1: Load all audio files in batches
        audio_data = {}
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            print(
                f"[Classifier] Loading audio batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}"
            )

            for file in batch_files:
                print(f"[Classifier] Loading audio for {file}")
                try:
                    loader = MonoLoader(
                        filename=file, sampleRate=self.sr, resampleQuality=self.rq
                    )
                    audio_data[file] = loader()
                except Exception as e:
                    if "File name too long" in str(e):
                        print(f"[Classifier] File name too long, using temp file for {file}")
                        try:
                            # Create temp file with same extension
                            ext = os.path.splitext(file)[1]
                            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                tmp_path = tmp.name
                                print(f"[Classifier] Using temp file {tmp_path}")
                            
                            # Use lock to safely change directory and copy using bytes
                            with dir_lock:
                                cwd = os.getcwd()
                                try:
                                    dirname = os.path.dirname(file)
                                    basename = os.path.basename(file)
                                    os.chdir(dirname)
                                    # Use os.fsencode on basename to handle bytes and avoid length limit
                                    shutil.copy2(os.fsencode(basename), tmp_path)
                                    print(f"[Classifier] Copied {basename} to {tmp_path}")
                                finally:
                                    os.chdir(cwd)
                            
                            # Load from temp path
                            loader = MonoLoader(
                                filename=tmp_path, sampleRate=self.sr, resampleQuality=self.rq
                            )
                            audio_data[file] = loader()
                            print(f"[Classifier] Loaded audio for {file}")
                            
                            # Clean up
                            os.remove(tmp_path)
                        except Exception as inner_e:
                            print(f"[Classifier] Error processing temp file for {file}: {str(inner_e)}")
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    else:
                        print(f"[Classifier] Error loading {file}: {str(e)}")

        if not audio_data:
            return results

        # Phase 2: Generate embeddings for all files
        print(f"[Classifier] Generating embeddings for {len(audio_data)} files")
        discogs_embeddings = {}

        for file, audio in audio_data.items():
            try:
                print(f"[Classifier] Generating embeddings for {file}")
                # Extract embeddings using Essentia models
                discogs_embeddings[file] = self.discogs_model(audio)
            except Exception as e:
                print(f"[Classifier] Error generating embeddings for {file}: {str(e)}")

        # Free memory for audio data
        del audio_data

        if not discogs_embeddings:
            return results

        # Phase 3: Process embeddings
        self._process_embeddings(discogs_embeddings, results)

        return results

    def _process_embeddings(self, discogs_embeddings, results):
        """Process embeddings sequentially"""
        for file, embedding in discogs_embeddings.items():
            try:
                data = {}
                # Use standard model calls
                data["genre"] = self._process_prediction(
                    self.genre_model,
                    "./classifier/models/genre_discogs400-discogs-effnet-1",
                    embedding,
                    threshold=0,
                )
                data["party"] = self._process_prediction(
                    self.party_model,
                    "./classifier/models/mood_party-discogs-effnet-1",
                    embedding,
                )
                data["relaxed"] = self._process_prediction(
                    self.relaxed_model,
                    "./classifier/models/mood_relaxed-discogs-effnet-1",
                    embedding,
                )
                data["danceability"] = self._process_prediction(
                    self.dance_model,
                    "./classifier/models/danceability-discogs-effnet-1",
                    embedding,
                )
                data["moodtheme"] = self._process_prediction(
                    self.moodtheme_model,
                    "./classifier/models/mtg_jamendo_moodtheme-discogs-effnet-1",
                    embedding,
                )
                
                if not data:
                    print(f"[Classifier] Warning: No data generated for {file}")
                else:
                    print(f"[Classifier] Generated {len(data)} classification types for {file}")

                results[file] = data
            except Exception as e:
                print(f"[Classifier] Error processing {file}: {str(e)}")

    def _post_process_prediction(self, predictions, model_name, threshold=0.10):
        """Post-process raw prediction results (without model inference)"""
        labels = self._get_labels(model_name)

        # Ensure predictions have correct shape
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)

        # Take mean across frames/segments (first dimension)
        mean = np.mean(predictions, axis=0)

        # Create result dictionary with descending scores
        res = {}
        for i, l in enumerate(mean.argsort()[0:][::-1], 1):
            if l < len(labels) and mean[l] >= threshold:
                res[labels[l]] = float(mean[l])

        return res

    def _process_prediction(self, model, model_name, embeddings, threshold=0.10):
        """Original method for non-batched processing"""
        labels = self._get_labels(model_name)
        predictions = model(embeddings)

        # Ensure predictions have correct shape
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)

        # Take mean across frames/segments (first dimension)
        mean = np.mean(predictions, axis=0)

        # Create result dictionary with descending scores
        res = {}
        for i, l in enumerate(mean.argsort()[0:][::-1], 1):
            if l < len(labels) and mean[l] >= threshold:
                res[labels[l]] = float(mean[l])

        return res

    # Legacy methods kept for compatibility
    def print_genre_predictions(self, model_name: str, embeddings):
        """Legacy method kept for compatibility"""
        labels = self._get_labels(model_name)
        model = TensorflowPredict2D(
            graphFilename="{}.pb".format(model_name),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        predictions = model(embeddings)
        mean = np.mean(predictions, axis=0)
        res = {}

        for i, l in enumerate(mean.argsort()[0:][::-1], 1):
            res[labels[l]] = mean[l]
        return res

    def print_mirex_predictions(self, model_name: str, embeddings):
        """Get MIREX mood predictions for the audio."""
        labels = self._get_labels(model_name)
        model = TensorflowPredict2D(
            graphFilename="./classifier/models/moods_mirex-msd-musicnn-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall",
        )
        predictions = model(embeddings)
        mean = np.mean(predictions, axis=0)
        res = {}

        for i, l in enumerate(mean.argsort()[0:][::-1], 1):
            res[labels[l]] = mean[l]
        return res

    def print_other_predictions(self, model_name: str, embeddings, o="model/Softmax"):
        """Legacy method kept for compatibility"""
        labels = self._get_labels(model_name)
        if o != "":
            model = TensorflowPredict2D(
                graphFilename="{}.pb".format(model_name), output=o
            )
        else:
            model = TensorflowPredict2D(graphFilename="{}.pb".format(model_name))
        predictions = model(embeddings)
        mean = np.mean(predictions, axis=0)
        res = {}

        for i, l in enumerate(mean.argsort()[0:][::-1], 1):
            if mean[l] < 0.10:
                break
            res[labels[l]] = mean[l]
        return res

