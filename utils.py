from siamese_network import generate_data, make_siamese_model, loss_and_opt, checkpoints, train, test, conf_gpu, preprocess, merge_data, split_data, create_embedding_model, create_distance_model
import os
import random
import numpy as np
import copy
import db_requests

def face_dataset(class_path, face_path):
    # Setup paths
    pos_anc = []
    positive = []
    negative = []
    anchor = []
    for face in os.listdir(class_path):
        if os.path.join(class_path, face) != os.path.join(class_path, face_path):
            for img in os.listdir(os.path.join(class_path, face)):
                negative.append(os.path.join(class_path, face, img))
        else:
            for img in os.listdir(os.path.join(class_path, face)):
                pos_anc.append(os.path.join(class_path, face, img))
    for i in range(9):
        random.shuffle(pos_anc)
        anchor = anchor + copy.deepcopy(pos_anc)
        random.shuffle(pos_anc)
        positive = positive + copy.deepcopy(pos_anc)
    data = generate_data(anchor, positive, negative)
    return data

# Create a new model and train it
def new_model(class_path, EPOCHS):
    model = make_siamese_model()
    # For debuging
    #model.run_eagerly = True
    binary_cross_loss, opt = loss_and_opt()
    checkpoint_prefix, checkpoint = checkpoints(opt, model)
    # Generate dataset
    data = 0
    for face in os.listdir(class_path):
        data_temp = face_dataset(class_path, face)
        data = merge_data(data, data_temp)
    train_data, test_data = split_data(data)
    # Train model
    train(train_data, EPOCHS, checkpoint,
          checkpoint_prefix, binary_cross_loss, opt, model)
    # Test model
    test(test_data, model)
    # Save weights
    model.save('siamesemodelv5.h5')

def load_model(path):
    model = make_siamese_model()
    model.load_weights(path)
    return model

# Compare two images using the model and returns True if similarity is above 0.5 else False 
def compare(model, input_img, anchor_img):
    result = model.predict(
        list(np.expand_dims([input_img, anchor_img], axis=1)))
    print(result)
    return result > 0.5

# Generate a vector of a given photo using the embedding model
def generate_vector(embedding_model, img_path):
    embedding_vector = embedding_model.predict(
        list(np.expand_dims([preprocess(img_path)], axis=1)))
    return embedding_vector


conf_gpu()
# def save_vectors(model, class_path, vector_class_path):
#     embedding_model = create_embedding_model(model)
#     vector_class = []
#     for face in os.listdir(class_path):
#         print(face)
#         vector_face = []
#         for img in os.listdir(os.path.join(class_path, face)):
#             vector_face.append(generate_vector(
#                 embedding_model, os.path.join(class_path, face, img)))
#         vector_class.append(vector_face)
#     save_vector_class(vector_class_path, vector_class)


#new_model('C:\\Users\\admin\\Downloads\\faces', 50)
# model = load_model('siamesemodelv5.h5')
# data = 0
# for face in os.listdir('C:\\Users\\admin\\Downloads\\faces'):
#     data_temp = face_dataset('C:\\Users\\admin\\Downloads\\faces', face)
#     data = merge_data(data, data_temp)
# train_data, test_data = split_data(data)

# test(test_data, model)

# # save_vectors(model, 'C:\\Users\\admin\\Downloads\\10 people', '10 people')
# # vector_class = load_vector_class('10 people')
# # print(len(vector_class))
# # print(len(vector_class[0]))

# embedding_model = create_embedding_model(model)
# vector = generate_vector(embedding_model, 'C:\\Users\\admin\\Downloads\\faces\\4\\Gerhard_Schroeder_0081.jpg')
# print(vector)
# # # distance_model = create_distance_model(model)
# curr_vector = get_vector(embedding_model, 'C:\\Users\\admin\\Downloads\\faces\\4\\Gerhard_Schroeder_0081.jpg')
# print(curr_vector)
# db_requests.set_vector('Gerhard_Schroeder', curr_vector)
# input_img = 'C:\\Users\\admin\\Downloads\\new_faces_cropped\\Bill_Clinton\\Bill_Clinton_0005.jpg'
# anchor_img = 'C:\\Users\\admin\\Downloads\\new_faces_cropped\\Angelina_Jolie\\Angelina_Jolie_0005.jpg'
# y = compare(model, input_img, anchor_img)
# print(y)
# # train model
# train(train_data, EPOCHS, checkpoint,
#       checkpoint_prefix, binary_cross_loss, opt, model)
# # test model
# test(test_data, model)
# # Save weights
# model.save('siamesemodelv2.h5')
# return checkpoint, checkpoint_prefix, binary_cross_loss, opt, model
