import os
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.python.keras.metrics import Precision, Recall
import tensorflow as tf


# Allows Tensorflow to use the GPU
def conf_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load the image from the path and preprocess it
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    # Resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0
    # Return image
    return img

# Preprocess the input data
def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

# Merges two datasets to one dataset
def merge_data(data_a, data_b):
    if data_a == 0:
        return data_b
    return data_a.concatenate(data_b)

# Create pairs of negative and positive with anchor and a lavel
def generate_data(anchor, positive, negative):
    # List to Tensorflow dataset
    anchor = tf.data.Dataset.from_tensor_slices(anchor)
    positive = tf.data.Dataset.from_tensor_slices(positive)
    negative = tf.data.Dataset.from_tensor_slices(negative)
    # Create a tuple like object with anchor, positive/negative/, label(1/0)
    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)
    return data

# Split data to train and test
def split_data(data):
    # Build dataloader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)
    # Training partition
    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)
    # Testing partition
    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)
    return train_data, test_data

# Build the embedding layers of the model
def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Build siamese model
def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))
    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    # Embedding layers
    embedding = make_embedding()

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),
                              embedding(validation_image))
    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# Define loss function and optimizer
def loss_and_opt():
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001
    return binary_cross_loss, opt


# Establish Checkpoints
def checkpoints(opt, siamese_model):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    return checkpoint_prefix, checkpoint

# Calculate loss of current batch
@tf.function
def train_step(batch, binary_cross_loss, opt, siamese_model):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    # Return loss
    return loss

# Train the model
def train(data, EPOCHS, checkpoint, checkpoint_prefix, binary_cross_loss, opt, siamese_model):
    x_epochs = []
    y_loss = []
    y_recall = []
    y_precision = []
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        x_epochs.append(epoch)
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch, binary_cross_loss, opt, siamese_model)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        y_loss.append(loss.numpy())
        y_recall.append(r.result().numpy())
        y_precision.append(p.result().numpy())
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    # Plot results in graph
    plt.plot(x_epochs, y_loss)
    plt.plot(x_epochs, y_recall)
    plt.plot(x_epochs, y_precision)
    plt.xlabel("epochs")
    plt.show()

# Test model
def test(test_data, siamese_model):
    # Calculate Metrics
    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        yhat_ = np.reshape(yhat, [-1])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
    print('recall\tprecision')
    print(r.result().numpy(), p.result().numpy())

def create_embedding_model(model):
    embedding_model = Model(
        inputs=model.input[0], outputs=model.get_layer("distance").input[0])
    return embedding_model

def create_distance_model(model):
    input_vector = Input(name='input_vector', shape=(4096))
    validation_vector = Input(name='validation_vector', shape=(4096))
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(input_vector,
                              validation_vector)
    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)
    distance_model = Model(
        inputs=[input_vector, validation_vector], outputs=classifier, name='distanceModel')
    distance_model.layers[2].set_weights(model.layers[3].get_weights())
    distance_model.layers[3].set_weights(model.layers[4].get_weights())
    return distance_model

# def save_vector_class(vector_class_path, vector_class):
#     # Create a dictionary to store the tensors and their shapes
#     dict = {}
#     for i, tensor in enumerate(vector_class):
#         tensor = tf.stack(tensor)
#         dict[f"tensor_{i}"] = {
#             "data": tensor.numpy().tolist(),
#             "shape": tensor.shape.as_list()
#         }

#     # Write the dictionary to a JSON file
#     with open(vector_class_path + '.json', "w") as f:
#         json.dump(dict, f)

# def save_vector_class(vector_class_path, vector_class):
#     for i in range(len(vector_class)):
#         vectors = tf.data.Dataset.from_tensor_slices(vector_class[i])
#         tf.data.Dataset.save(vectors, vector_class_path + '/' + str(i))


# def load_vector_class(vector_class_path):
#     # get the number of vectors in the vector class
#     num_vectors = len(os.listdir(vector_class_path))

#     # create an empty list to hold the vectors
#     vector_class = []

#     # iterate over the vectors and read them from the files
#     for i in range(num_vectors):
#         vector_path = os.path.join(vector_class_path, str(i))
#         vectors = tf.data.Dataset.load(vector_path)
#         vector_class.append(vectors)

#     return vector_class

# def load_vector_class(vector_class_path):
#     # Read the dictionary from the JSON file
#     with open(vector_class_path + '.json', "r") as f:
#         my_dict = json.load(f)

    # Reconstruct the list of tensors
    # vector_class = []
    # for i in range(len(my_dict)):
    #     tensor_data = my_dict[f"tensor_{i}"]["data"]
    #     tensor_shape = my_dict[f"tensor_{i}"]["shape"]
    #     tensor = tf.reshape(tf.constant(tensor_data), tensor_shape)
    #     vector_class.append(tensor)

    # return vector_class
