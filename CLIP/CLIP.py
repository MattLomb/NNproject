from CLIP.DatasetLoader import DatasetLoader

import os
import collections
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")


def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


def create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    # Load the pre-trained Xception model to be used as the base encoder.
    xception = keras.applications.Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in xception.layers:
        layer.trainable = trainable
    # Receive the images as inputs.
    inputs = layers.Input(shape=(218, 218, 3), name="image_input")
    # Preprocess the input image.
    xception_input = tf.keras.applications.xception.preprocess_input(inputs)
    # Generate the embeddings for the images using the xception model.
    embeddings = xception(xception_input)
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(inputs, outputs, name="vision_encoder")


def create_text_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    # Load the BERT preprocessing module.
    preprocess = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
        name="text_preprocessing",
    )
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
        "bert",
    )
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
    # Preprocess the text.
    bert_inputs = preprocess(inputs)
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(bert_inputs)["pooled_output"]
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")


class DualEncoder(keras.Model):
    def __init__(self, text_encoder, image_encoder, temperature=1.0, **kwargs):
        super(DualEncoder, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        # Place each encoder on a separate GPU (if available).
        # TF will fallback on available devices if there are fewer than 2 GPUs.
        with tf.device("/gpu:0"):
            # Get the embeddings for the captions.
            caption_embeddings = self.text_encoder(features["caption"], training=training)
        with tf.device("/gpu:1"):
            # Get the embeddings for the images.
            image_embeddings = self.image_encoder(features["image"], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        logits = (
                tf.matmul(caption_embeddings, image_embeddings, transpose_b=True)
                / self.temperature
        )
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = tf.matmul(
            image_embeddings, image_embeddings, transpose_b=True
        )
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = tf.matmul(
            caption_embeddings, caption_embeddings, transpose_b=True
        )
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        # Compute the loss for the captions using crossentropy
        captions_loss = keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        # Compute the loss for the images using crossentropy
        images_loss = keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}





class CLIP:

    def __init__(self, load_weights=False):
        self.vision_encoder = create_vision_encoder(
            num_projection_layers=1, projection_dims=256, dropout_rate=0.1
        )
        self.text_encoder = create_text_encoder(
            num_projection_layers=1, projection_dims=256, dropout_rate=0.1
        )

        if load_weights:
            self.vision_encoder.load_weights("CLIP/weights/vision")
            self.text_encoder.load_weights("CLIP/weights/text")

        self.dual_encoder = DualEncoder(self.text_encoder, self.vision_encoder, temperature=0.05)
        self.dual_encoder.compile(
            optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001),
            metrics=[tf.keras.metrics.Accuracy()]
        )

    def train(self, epoches, batch_size):
        print("loading dataset")
        dataset_loader = DatasetLoader()
        ds = dataset_loader.getDataset(batch_size)

        dataset_size = dataset_loader.count()
        train_size = round(dataset_size * (2 / 3))

        train_ds = ds.take(train_size)
        validation_ds = ds.skip(train_size)

        print("Train dataset size", train_size, "Validation dataset size", dataset_size - train_size)
        print("Steps per epoch", train_size // batch_size)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )
        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        print("Starting train")
        history = self.dual_encoder.fit(
            train_ds,
            epochs=epoches,
            validation_data=validation_ds,
            callbacks=[reduce_lr, early_stopping],
        )

        print("Training completed. Saving vision and text encoders...")
        self.vision_encoder.save_weights("CLIP/weights/vision")
        self.text_encoder.save_weights("CLIP/weights/text")
        print("Models are saved.")

    # A string
    def predict_text(self, text):
        text = text.lower()
        return self.text_encoder.predict(tf.convert_to_tensor([text]))

    def predict_image(self, image):
        return self.vision_encoder.predict(image)
