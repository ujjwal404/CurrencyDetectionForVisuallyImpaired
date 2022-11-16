import tensorflow as tf


def convert_to_tflite(saved_model_dir, tflite_filename="model2.tflite"):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
