import tensorflow as tf
import os

def convert_model(h5_path):
    model_name = os.path.splitext(os.path.basename(h5_path))[0]
    tflite_path = os.path.join(os.path.dirname(h5_path), model_name + ".tflite")

    if os.path.exists(tflite_path):
        print(f"✔ Ya existe: {tflite_path}")
        return

    print(f"🔄 Convirtiendo: {h5_path}")

    # Cargar modelo Keras
    try:
        model = tf.keras.models.load_model(h5_path)

        # Convertir a TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimización opcional
        tflite_model = converter.convert()

        # Guardar archivo .tflite
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"✅ Guardado: {tflite_path}")

    except Exception as e:
        print(f"❌ Error al convertir {h5_path}: {e}")

def convert_all_in_directory(root_dir):
    print(f"🔍 Buscando modelos en: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".h5"):
                h5_full_path = os.path.join(dirpath, filename)
                convert_model(h5_full_path)

if __name__ == "__main__":
    # Cambia esto si usas otro nombre de carpeta
    root_model_dir = "./"
    convert_all_in_directory(root_model_dir)
