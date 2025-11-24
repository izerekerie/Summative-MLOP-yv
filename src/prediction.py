"""
Prediction logic for waste classification.
Loads model and makes predictions on preprocessed images.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import tensorflow as tf

from preprocessing import DEFAULT_IMG_SIZE


def load_model_config(config_path: str = None) -> Dict:
    """
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to model_config.json. If None, searches in common locations.
    
    Returns:
        Dictionary with model configuration
    """
    if config_path is None:
        # Try to find config in common locations
        possible_paths = [
            Path(__file__).parent.parent / 'models' / 'model_config.json',
            Path(__file__).parent.parent / 'notebook' / 'models' / 'model_config.json',
            Path('/content/drive/MyDrive/waste-classification/models/model_config.json'),
            Path('models/model_config.json'),
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "Could not find model_config.json. Please specify the path or ensure "
                "the model has been trained and config file exists."
            )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def _read_h5_config(model_path: str) -> Dict:
    """Load serialized Keras model config from an H5 file."""
    with h5py.File(model_path, "r") as h5_file:
        config = h5_file.attrs.get("model_config")
        if config is None:
            raise ValueError(
                "Model file does not contain architecture configuration."
            )
        if isinstance(config, bytes):
            config = config.decode("utf-8")
        return json.loads(config)


def _extract_metadata_from_h5(model_path: str) -> Dict:
    """
    Read model metadata (input size, channels, classes, etc.) from an H5 file.
    """
    config = _read_h5_config(model_path)
    layers = config.get("config", {}).get("layers", [])

    input_layer = next(
        (layer for layer in layers if layer["class_name"] == "InputLayer"), None
    )
    if not input_layer:
        raise ValueError(
            "Could not locate InputLayer metadata inside the model file."
        )

    batch_shape = input_layer["config"].get("batch_shape")
    if not batch_shape or len(batch_shape) != 4:
        raise ValueError(f"Unexpected input batch shape metadata: {batch_shape}")

    height, width, channels = batch_shape[1], batch_shape[2], batch_shape[3]
    classifier_layer = next(
        (layer for layer in layers if layer["config"].get("name") == "classifier"),
        None,
    )
    if not classifier_layer:
        raise ValueError(
            "Could not locate classifier layer metadata to determine num_classes."
        )

    num_classes = classifier_layer["config"].get("units")
    kernel_reg = classifier_layer["config"].get("kernel_regularizer", {})
    l2_weight = kernel_reg.get("config", {}).get("l2", 0.0)

    dropout_layer = next(
        (layer for layer in layers if layer["config"].get("name") == "dropout"), None
    )
    dropout_rate = dropout_layer["config"].get("rate", 0.2) if dropout_layer else 0.2

    return {
        "img_size": (int(height), int(width)),
        "channels": int(channels),
        "num_classes": int(num_classes),
        "dropout_rate": float(dropout_rate),
        "l2_weight": float(l2_weight),
    }


def _build_model_from_metadata(metadata: Dict) -> tf.keras.Model:
    """Rebuild the EfficientNetB0 classifier based on saved metadata."""
    img_size: Tuple[int, int] = metadata["img_size"]
    channels = metadata["channels"]
    if channels != 3:
        raise ValueError(
            "Saved model uses "
            f"{channels} input channels; automatic rebuild expects RGB."
        )

    num_classes = metadata["num_classes"]
    dropout_rate = metadata.get("dropout_rate", 0.2)
    l2_weight = metadata.get("l2_weight", 1e-4)

    inputs = tf.keras.layers.Input(shape=img_size + (channels,), name="input_image")
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )
    augmented = data_augmentation(inputs)
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    normalized = tf.keras.layers.Lambda(
        lambda tensor: preprocess(tensor), name="preprocess_input"
    )(augmented)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=normalized,
        weights=None,
    )
    base_model.trainable = False

    pooled = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(
        base_model.output
    )
    dropped = tf.keras.layers.Dropout(dropout_rate, name="dropout")(pooled)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
        name="classifier",
    )(dropped)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_waste")


def _load_model_via_weights(model_path: str) -> tf.keras.Model:
    """Rebuild the architecture and load weights manually from an H5 file."""
    metadata = _extract_metadata_from_h5(model_path)
    rebuilt_model = _build_model_from_metadata(metadata)
    rebuilt_model.load_weights(model_path)
    print(
        "Model architecture rebuilt from metadata and weights loaded successfully. "
        f"Input shape restored as {rebuilt_model.input_shape}."
    )
    return rebuilt_model


def load_model(model_path: str = None) -> tf.keras.Model:
    """
    Load the trained TensorFlow model from /models/waste_classifier_final.keras.
    
    Args:
        model_path: Path to saved model. If None, uses path from config or default.
    
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        # Try to get from config
        try:
            config = load_model_config()
            config_model_path = config.get('model_path')
            
            # Check if config path exists (might be Colab path)
            if config_model_path and Path(config_model_path).exists():
                model_path = config_model_path
            # If config path doesn't exist, try to find local equivalent
            elif config_model_path:
                # Extract filename from config path
                config_filename = Path(config_model_path).name
                # Try to find it in local models directory
                local_path = Path(__file__).parent.parent / 'models' / config_filename
                if local_path.exists():
                    model_path = str(local_path)
                    print(f"Config path not found, using local: {model_path}")
                # If .h5 file, try .keras version instead
                elif config_filename.endswith('.h5'):
                    keras_filename = config_filename.replace('.h5', '.keras')
                    keras_path = Path(__file__).parent.parent / 'models' / keras_filename
                    if keras_path.exists():
                        model_path = str(keras_path)
                        print(f"Config path not found, using local .keras file: {model_path}")
        except FileNotFoundError:
            pass
        
        if not model_path or not Path(model_path).exists():
            searched_paths = [
                str(Path(__file__).parent.parent / 'models'),
                'models',
            ]
            raise FileNotFoundError(
                f"Model not found. Searched in: {searched_paths}\n"
                "Please ensure:\n"
                "1. The model has been trained (run the notebook)\n"
                "2. The model is saved as 'waste_classifier_final.keras'\n"
                "3. The model is in the 'models/' directory"
            )
    
    print(f"Loading model from: {model_path}")
    load_errors = []
    try:
        # Try loading with compile=False first to avoid validation issues
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully (without compilation)!")
        except Exception as error_compile_false:
            load_errors.append(error_compile_false)
            print(f"Loading without compilation failed: {error_compile_false}")
            print("Trying to load with compilation...")
            model = tf.keras.models.load_model(model_path, compile=True)
            print("Model loaded successfully with compilation!")
    except Exception as error_compile_true:
        load_errors.append(error_compile_true)
        mismatch_error = None
        for err in load_errors:
            err_message = str(err).lower()
            if "stem_conv" in err_message and "shape mismatch" in err_message:
                mismatch_error = err
                break
        if mismatch_error:
            print(
                "Detected EfficientNet stem_conv channel mismatch while loading the "
                "saved model. Rebuilding architecture and loading weights manually..."
            )
            try:
                model = _load_model_via_weights(model_path)
            except Exception as rebuild_error:
                raise RuntimeError(
                    "Automatic rebuild attempt failed after encountering a channel "
                    "mismatch:\n"
                    f" - Original error: {mismatch_error}\n"
                    f" - Rebuild error: {rebuild_error}"
                ) from rebuild_error
        else:
            raise RuntimeError(
                f"Failed to load model from {model_path}: {error_compile_true}\n"
                "Make sure the model file is valid and in the correct format."
            ) from error_compile_true

    # Verify input shape expects 3 channels
    input_shape = model.input_shape
    if input_shape and len(input_shape) == 4:
        expected_channels = input_shape[-1]
        if expected_channels != 3:
            print(
                "Warning: Model expects "
                f"{expected_channels} channels, but preprocessing provides 3 channels"
            )
        else:
            print(f"Model input shape verified: {input_shape}")

    # Recompile the model if needed (optional, for metrics)
    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("Model compiled successfully!")
    except Exception as compile_error:
        print(f"Warning: Could not compile model (non-critical): {compile_error}")
        print("Model will work for inference without compilation.")
    
    return model


def get_class_names(config_path: str = None) -> List[str]:
    """
    Get class names from config file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        List of class names
    """
    config = load_model_config(config_path)
    return config.get('class_names', [])


def get_img_size(config_path: str = None) -> tuple:
    """
    Get image size from config file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Image size tuple (height, width)
    """
    config = load_model_config(config_path)
    return tuple(config.get('img_size', DEFAULT_IMG_SIZE))


def predict_image(model: tf.keras.Model, image_bytes: bytes, 
                 class_names: List[str], img_size: tuple = DEFAULT_IMG_SIZE) -> Dict:
    """
    Make prediction on a single image.
    
    Args:
        model: Loaded Keras model
        image_bytes: Image file bytes
        class_names: List of class names
        img_size: Image size (height, width)
    
    Returns:
        Dictionary with prediction results
    """
    from preprocessing import preprocess_image
    
    # Preprocess image
    input_tensor = preprocess_image(image_bytes, img_size)
    
    # Validate input shape
    if len(input_tensor.shape) != 4:
        raise ValueError(f"Invalid input tensor shape: {input_tensor.shape}. Expected (1, H, W, 3).")
    if input_tensor.shape[0] != 1:
        raise ValueError(f"Invalid batch size: {input_tensor.shape[0]}. Expected 1.")
    if input_tensor.shape[3] != 3:
        raise ValueError(f"Invalid number of channels: {input_tensor.shape[3]}. Expected 3 (RGB).")
    
    # Make prediction
    predictions = model.predict(input_tensor, verbose=0)
    
    # Get predicted class
    predicted_idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    
    # Create response
    return {
        "label": class_names[predicted_idx],
        "confidence": confidence,
        "probabilities": {
            cls: float(prob) for cls, prob in zip(class_names, predictions[0])
        }
    }


def predict_image_from_path(model: tf.keras.Model, image_path: str,
                            class_names: List[str], img_size: tuple = DEFAULT_IMG_SIZE) -> Dict:
    """
    Make prediction on an image from file path.
    
    Args:
        model: Loaded Keras model
        image_path: Path to image file
        class_names: List of class names
        img_size: Image size (height, width)
    
    Returns:
        Dictionary with prediction results
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    return predict_image(model, image_bytes, class_names, img_size)


def predict_batch(model: tf.keras.Model, images_data: List[bytes],
                 class_names: List[str], img_size: tuple = DEFAULT_IMG_SIZE) -> List[Dict]:
    """
    Make predictions on multiple images.
    
    Args:
        model: Loaded Keras model
        images_data: List of image bytes
        class_names: List of class names
        img_size: Image size (height, width)
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image_bytes in images_data:
        try:
            result = predict_image(model, image_bytes, class_names, img_size)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return results
