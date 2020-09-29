import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
import glob
import random
import os
BATCH_SIZE = 32
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def load_data():
    global all_images, all_label, name_label_type, color_label_type, name_labels, color_labels
    all_images = glob.glob('./dataset/*/*')
    random.shuffle(all_images)
    all_labels = [item.split('/')[2] for item in all_images]
    name_labels = [item.split('_')[0] for item in all_labels]
    color_labels = [item.split('_')[1] for item in all_labels]
    name_label_type = set(item for item in name_labels)
    color_label_type = set(item for item in color_labels)
    name_lable_index = dict((name, index) for index, name in enumerate(name_label_type))
    color_lable_index = dict((color, index) for index, color in enumerate(color_label_type))
    print(name_lable_index)
    print(len(name_lable_index))
    print(color_lable_index)
    print(len(color_lable_index))
    name_labels = [name_lable_index[name] for name in name_labels]
    color_labels = [color_lable_index[color] for color in color_labels]




def load_preprocess_images(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = 2 * image - 1
    return image


def to_dataset():
    global train_dataset, valid_dataset, train_num
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_dataset = tf.data.Dataset.from_tensor_slices(all_images)
    image_dataset = path_dataset.map(load_preprocess_images, num_parallel_calls=AUTOTUNE)
    label_dataset = tf.data.Dataset.from_tensor_slices((color_labels, name_labels))
    multi_pred_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    all_images_num = len(all_images)
    train_num = int(all_images_num * 0.8)
    train_dataset = multi_pred_dataset.take(train_num).repeat().shuffle(train_num).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    valid_dataset = multi_pred_dataset.skip(train_num).batch(BATCH_SIZE)

def train():
    model = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
    ])
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Dense(1024, activation='relu')(x)
    x1 = layers.Dropout(0.5)(x1)
    out_color = layers.Dense(len(color_label_type), activation='softmax', name='out_color')(x1)

    x2 = layers.Dense(1024, activation='relu')(x)
    x2 = layers.Dropout(0.5)(x2)
    out_name = layers.Dense(len(name_label_type), activation='softmax', name='out_name')(x2)

    model = tf.keras.Model(inputs=inputs, outputs=[out_color, out_name])
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(lr=0.0001),
        loss={
            'out_color': 'sparse_categorical_crossentropy',
            'out_name': 'sparse_categorical_crossentropy',
        },
        metrics=['acc']
    )
    model.fit(
        train_dataset,
        epochs=15,
        steps_per_epoch=train_num // BATCH_SIZE,
        validation_data=valid_dataset,
        validation_steps=1
    )
    model.save('multi_pred.h5')


if __name__ == '__main__':
    load_data()
    to_dataset()
    train()