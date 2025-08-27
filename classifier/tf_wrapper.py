import tensorflow as tf
import numpy as np
import os
import time

# Force TensorFlow to use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Global shared session for all models
_global_session = None

def get_global_session():
    """Returns a singleton global TensorFlow session"""
    global _global_session
    if _global_session is None:
        # Configure session for GPU usage
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True,
                                            force_gpu_compatible=True)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                        log_device_placement=False,
                                        allow_soft_placement=True)
        config.gpu_options.visible_device_list = "0"  # Use first GPU

        # Create session
        _global_session = tf.compat.v1.Session(config=config)
        print("[GPUWrapper] Created global TensorFlow session for GPU")

        # Create and run a simple test operation to ensure GPU is working
        test_graph = tf.Graph()
        with test_graph.as_default():
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)

            # We need to use a session with this specific graph
            test_sess = tf.compat.v1.Session(graph=test_graph, config=config)
            start_time = time.time()
            test_sess.run(c)
            end_time = time.time()
            print(f"[GPUWrapper] GPU test completed in {end_time - start_time:.4f} seconds")
            test_sess.close()

    return _global_session

class GPUModelWrapper:
    """A wrapper that forces TensorFlow models to run on GPU with compatible output format"""

    def __init__(self, model_path, input_name=None, output_name=None):
        """Initialize with model path and optional tensor names"""
        print(f"[GPUWrapper] Loading model {model_path} for GPU")
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

        # Load the graph definition from the frozen model
        with tf.io.gfile.GFile(model_path, "rb") as f:
            self.graph_def = tf.compat.v1.GraphDef()
            self.graph_def.ParseFromString(f.read())

        # Import the graph definition into a new graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")

            # Create a dedicated session for this model
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
            self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        # Get input and output tensor names
        self.input_name = self._find_input_tensor() if input_name is None else input_name
        self.output_name = self._find_output_tensor() if output_name is None else output_name

        print(f"[GPUWrapper] Model ready with input: {self.input_name}, output: {self.output_name}")

    def _find_input_tensor(self):
        """Find the input tensor name in the graph"""
        with self.graph.as_default():
            operations = self.graph.get_operations()

            # Common input tensor names
            common_inputs = ["Placeholder:0", "input:0", "serving_default_model_Placeholder:0"]

            for name in common_inputs:
                try:
                    self.graph.get_tensor_by_name(name)
                    return name
                except:
                    continue

            # Look for placeholder operations
            for op in operations:
                if op.type == "Placeholder":
                    return op.outputs[0].name

            return operations[0].outputs[0].name

    def _find_output_tensor(self):
        """Find the output tensor name in the graph"""
        with self.graph.as_default():
            # Map of known model files to their output tensor names
            known_outputs = {
                "discogs-effnet-bs64-1.pb": "PartitionedCall:1",
                "msd-musicnn-1.pb": "model/dense/BiasAdd:0",
                "genre_discogs400-discogs-effnet-1.pb": "PartitionedCall:0",
                "mood_party-discogs-effnet-1.pb": "model/Softmax:0",
                "mood_relaxed-discogs-effnet-1.pb": "model/Softmax:0",
                "danceability-discogs-effnet-1.pb": "model/Softmax:0"
            }

            if self.model_name in known_outputs:
                try:
                    self.graph.get_tensor_by_name(known_outputs[self.model_name])
                    return known_outputs[self.model_name]
                except:
                    pass

            # Try common output names
            common_outputs = ["output:0", "model/Softmax:0", "PartitionedCall:0", "PartitionedCall:1"]
            for name in common_outputs:
                try:
                    self.graph.get_tensor_by_name(name)
                    return name
                except:
                    continue

            operations = self.graph.get_operations()
            return operations[-1].outputs[0].name

    def batch_predict(self, embeddings_list):
        """Run inference on a batch of embeddings to minimize transfers"""
        if not embeddings_list:
            return []

        try:
            # Stack embeddings into a single batch if possible
            if isinstance(embeddings_list[0], np.ndarray) and all(e.shape == embeddings_list[0].shape for e in embeddings_list):
                # If all embeddings have the same shape, stack them into one batch
                stacked_embeddings = np.vstack(embeddings_list)

                # Run inference on the batch
                with self.graph.as_default():
                    with tf.device('/GPU:0'):
                        input_tensor = self.graph.get_tensor_by_name(self.input_name)
                        output_tensor = self.graph.get_tensor_by_name(self.output_name)

                        start_time = time.time()
                        results = self.sess.run(output_tensor, {input_tensor: stacked_embeddings})
                        end_time = time.time()

                # Split results back into individual predictions
                batch_size = len(embeddings_list)
                split_results = np.array_split(results, batch_size, axis=0)

                print(f"[GPUWrapper] Batch inference for {batch_size} items took {end_time - start_time:.4f}s")
                return split_results
            else:
                # If embeddings have different shapes, process them individually
                print("[GPUWrapper] Cannot batch - embeddings have different shapes")
                return [self(embedding) for embedding in embeddings_list]

        except Exception as e:
            print(f"[GPUWrapper] Batch prediction error: {e}")
            # Fall back to individual processing
            return [self(embedding) for embedding in embeddings_list]

    def __call__(self, audio):
        """Run inference with the model on GPU and format output to match Essentia"""
        # Ensure audio is numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        try:
            # Use the model's dedicated session with its graph
            with self.graph.as_default():
                with tf.device('/GPU:0'):
                    input_tensor = self.graph.get_tensor_by_name(self.input_name)
                    output_tensor = self.graph.get_tensor_by_name(self.output_name)
                    result = self.sess.run(output_tensor, {input_tensor: audio})

            # Format the result to match Essentia's output format
            if len(result.shape) == 1:
                result = np.expand_dims(result, axis=0)

            # Ensure consistent shape for specific model types
            if "genre_discogs400" in self.model_path or "mood" in self.model_path or "danceability" in self.model_path:
                if len(result.shape) > 2:
                    result = result.reshape(1, -1)

            return result

        except Exception as e:
            print(f"[GPUWrapper] GPU inference failed: {e}, falling back to CPU")
            # Fall back to CPU if GPU fails
            with self.graph.as_default():
                with tf.device('/CPU:0'):
                    input_tensor = self.graph.get_tensor_by_name(self.input_name)
                    output_tensor = self.graph.get_tensor_by_name(self.output_name)
                    result = self.sess.run(output_tensor, {input_tensor: audio})

                if len(result.shape) == 1:
                    result = np.expand_dims(result, axis=0)

                return result
