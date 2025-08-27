import numpy as np
import json
import tensorflow as tf
from classifier.tf_wrapper import GPUModelWrapper, get_global_session

# Configure TensorFlow
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[Classifier] Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from essentia.standard import (
    MonoLoader,  # pyright: ignore[reportAttributeAccessIssue]
    TensorflowPredictEffnetDiscogs,  # pyright: ignore[reportAttributeAccessIssue]
    TensorflowPredict2D,  # pyright: ignore[reportAttributeAccessIssue]
)


class Classifier:
    # Our models take audio streams at 16kHz
    sr = 16000
    rq = 4

    def __init__(self):
        """Initialize classifier with optimized GPU model loading"""
        # Initialize the shared global session
        if gpus:
            print("[Classifier] Initializing GPU session")
            _ = get_global_session()

        print("[Classifier] Loading primary embedding models with Essentia")
        self.discogs_model = TensorflowPredictEffnetDiscogs(
            graphFilename="./classifier/models/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1",
        )

        # For the secondary models that process embeddings, use GPU if available
        if gpus:
            print("[Classifier] Using GPU for embedding processing models")

            # Use our GPU wrapper for these models
            self.genre_model = GPUModelWrapper(
                "./classifier/models/genre_discogs400-discogs-effnet-1.pb",
                input_name="serving_default_model_Placeholder:0",
                output_name="PartitionedCall:0",
            )
            self.party_model = GPUModelWrapper(
                "./classifier/models/mood_party-discogs-effnet-1.pb",
                output_name="model/Softmax:0",
            )
            self.relaxed_model = GPUModelWrapper(
                "./classifier/models/mood_relaxed-discogs-effnet-1.pb",
                output_name="model/Softmax:0",
            )
            self.dance_model = GPUModelWrapper(
                "./classifier/models/danceability-discogs-effnet-1.pb",
                output_name="model/Softmax:0",
            )
            self.moodtheme_model = GPUModelWrapper(
                "./classifier/models/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
            )
        else:
            print("[Classifier] No GPU found, using CPU for all models")
            # Use Essentia for all models when no GPU is available
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
        """Process multiple audio files with optimized batch processing"""
        results = {}

        if not files:
            return results

        # Use larger batch size for better GPU utilization
        batch_size = 20  # Increased from 5
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
                    print(f"[Classifier] Error loading {file}: {str(e)}")

        if not audio_data:
            return results

        # Phase 2: Generate embeddings for all files
        print(f"[Classifier] Generating embeddings for {len(audio_data)} files")
        discogs_embeddings = {}
        # musicnn_embeddings = {}

        for file, audio in audio_data.items():
            try:
                print(f"[Classifier] Generating embeddings for {file}")
                # Extract embeddings using Essentia models
                discogs_embeddings[file] = self.discogs_model(audio)
                # musicnn_embeddings[file] = self.musicnn_model(audio)
            except Exception as e:
                print(f"[Classifier] Error generating embeddings for {file}: {str(e)}")

        # Free memory for audio data
        del audio_data

        if not discogs_embeddings:
            return results

        # Phase 3: Batch process embeddings through each model
        if gpus:
            # Process with GPU batching
            self._process_with_gpu_batching(discogs_embeddings, results)
        else:
            # Process sequentially with CPU
            self._process_with_cpu(discogs_embeddings, results)

        return results

    def _process_with_gpu_batching(self, discogs_embeddings, results):
        """Process all embeddings in batches with GPU acceleration"""
        # Prepare lists of files and embeddings
        files = list(discogs_embeddings.keys())
        embeddings_list = [discogs_embeddings[file] for file in files]

        # Process genre predictions in a batch
        print(f"[Classifier] Batch processing {len(files)} files for genre")
        genre_predictions = self.genre_model.batch_predict(embeddings_list)

        # Process mood/party predictions in a batch
        print(f"[Classifier] Batch processing {len(files)} files for party mood")
        party_predictions = self.party_model.batch_predict(embeddings_list)

        # Process relaxed predictions in a batch
        print(f"[Classifier] Batch processing {len(files)} files for relaxed mood")
        relaxed_predictions = self.relaxed_model.batch_predict(embeddings_list)

        # Process danceability predictions in a batch
        print(f"[Classifier] Batch processing {len(files)} files for danceability")
        dance_predictions = self.dance_model.batch_predict(embeddings_list)

        # Process mood/theme predictions in a batch
        print(f"[Classifier] Batch processing {len(files)} files for mood/theme")
        moodtheme_predictions = self.moodtheme_model.batch_predict(embeddings_list)

        # Process results for each file
        for i, file in enumerate(files):
            try:
                data = {}

                # Post-process each prediction type
                if i < len(genre_predictions):
                    data["genre"] = self._post_process_prediction(
                        genre_predictions[i],
                        "./classifier/models/genre_discogs400-discogs-effnet-1",
                        threshold=0,
                    )

                if i < len(party_predictions):
                    data["party"] = self._post_process_prediction(
                        party_predictions[i], "./classifier/models/mood_party-discogs-effnet-1"
                    )

                if i < len(relaxed_predictions):
                    data["relaxed"] = self._post_process_prediction(
                        relaxed_predictions[i],
                        "./classifier/models/mood_relaxed-discogs-effnet-1",
                    )

                if i < len(dance_predictions):
                    data["danceability"] = self._post_process_prediction(
                        dance_predictions[i],
                        "./classifier/models/danceability-discogs-effnet-1",
                    )

                if i < len(moodtheme_predictions):
                    data["moodtheme"] = self._post_process_prediction(
                        moodtheme_predictions[i],
                        "./classifier/models/mtg_jamendo_moodtheme-discogs-effnet-1",
                    )

                results[file] = data
            except Exception as e:
                print(f"[Classifier] Error processing results for {file}: {str(e)}")

    def _process_with_cpu(self, discogs_embeddings, results):
        """Process embeddings sequentially with CPU"""
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
