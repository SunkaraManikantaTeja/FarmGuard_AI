"""Train a MobileNetV2 baseline. Usage: python train.py path/to/PlantVillage  (one folder per class)."""
import sys, json, tensorflow as tf
d = sys.argv[1]
tr = tf.keras.utils.image_dataset_from_directory(d, validation_split=.2, subset="training", seed=1, image_size=(224, 224), batch_size=32)
va = tf.keras.utils.image_dataset_from_directory(d, validation_split=.2, subset="validation", seed=1, image_size=(224, 224), batch_size=32)
names = tr.class_names
base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet"); base.trainable = False
inp = tf.keras.Input((224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
x = base(x, training=False); x = tf.keras.layers.GlobalAveragePooling2D()(x); x = tf.keras.layers.Dropout(.2)(x)
out = tf.keras.layers.Dense(len(names), activation="softmax")(x)
m = tf.keras.Model(inp, out)
m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
m.fit(tr, validation_data=va, epochs=5)
m.save("models/model.keras"); json.dump(names, open("models/labels.json", "w"))
print("Report your own validation accuracy honestly from the fit() output above.")
