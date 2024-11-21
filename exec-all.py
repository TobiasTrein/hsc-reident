
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
    
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import random

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from itertools import product 

import visualkeras
from imgaug import augmenters as iaa

np.random.seed(2024)
IMG_SIZE    = 150
num_classes = 10
batch_size  = 16

## Helper function to create the base models
def get_base_model(b_model, input_shape):
    model_dict = {
        'vgg': tf.keras.applications.vgg16.VGG16,
        'efficient': tf.keras.applications.EfficientNetB1,
        'mobile': tf.keras.applications.MobileNet,
        'mobileV2': tf.keras.applications.MobileNetV2,
        'MobileNetV3Small': tf.keras.applications.MobileNetV3Small,
        'MobileNetV3Large': tf.keras.applications.MobileNetV3Large
    }

    # Retrieve the model class from the dictionary and instantiate it
    if b_model in model_dict:
        return model_dict[b_model](include_top=False, input_shape=input_shape, weights='imagenet')
    else:
        raise ValueError(f"Unsupported model: {b_model}")

def get_augmentation_pipeline(augmentation_type):
    if augmentation_type == "flip": 
        return iaa.Sequential([
            iaa.Fliplr(1) # flip horizontally
        ])
    elif augmentation_type == "rotate":
        return iaa.Sequential([
            iaa.Rotate((-20, 20))  # Rotate between -20 and 20 degrees
        ])
    if augmentation_type == "noise":
        return iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
        ])
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        
def make_train_data(photo_type, labels):
    for label in labels:
        
        folder_name = next((f for f in os.listdir(dir) if f.startswith(label)), None)
        if folder_name:
            subfolders = ['top', 'front'] if photo_type == 'all' else [photo_type]
            
            for subfolder in subfolders:
                DIR = os.path.join(dir, folder_name, subfolder)
                if os.path.exists(DIR):
                    for img_file in tqdm(os.listdir(DIR)):
                        path = os.path.join(DIR, img_file)
                        img = cv2.imread(path, cv2.IMREAD_COLOR)
                        if img is not None:
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            X.append(np.array(img))
                            Z.append(str(folder_name))
                else:
                    print(f"Directory {DIR} does not exist")

## Helper function to create positive pairs and negative pairs
def create_pairs(images, labels, numClasses=num_classes):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive (0) or negative (1)
    pairImages = []
    pairLabels = []
    
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    idx = [np.where(labels == i)[0] for i in range(numClasses)]
    
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current iteration
        currentImage = images[idxA]
        label = labels[idxA]
        
        # randomly pick on an image that belongs to the *same* class label
        posId = random.choice(idx[label])
        posImage = images[posId]
        
        # prepare a positive pair and update the images and labels
        pairImages.append([currentImage, posImage])
        pairLabels.append([0])
        
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negId = np.where(labels != label)[0]
        negIdx = random.choice(negId)
        negImage = images[negIdx]
        
        # prepare a negative pair of images and update out lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([1])
    
    return (np.array(pairImages), np.array(pairLabels))


# Function to calculate the distance between two images (Euclidean Distance used here)
import tensorflow.keras.backend as K
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

# contrastive loss function
def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean((1 - y) * squaredPreds + y * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss


# Base model with pre-training EfficientNet
def embedding_model_cl(inputShape, b_model, embeddingDim=128):

    base_model = get_base_model(b_model, inputShape)

    # freeze all the layers of EfficientNet, so they won't be trained.
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = tf.keras.layers.Input(shape=inputShape)

    x = base_model(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=embeddingDim)(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Complete model
def complete_model_cl(base_model, LR):
    # Create the complete model with pair
    # embedding models and minimize the distance for positive pair
    # and maximum the distance for negative pair
    # between their output embeddings
    imgA = tf.keras.layers.Input(shape=((IMG_SIZE, IMG_SIZE, 3)))
    imgB = tf.keras.layers.Input(shape=((IMG_SIZE, IMG_SIZE, 3)))
    
    featsA = base_model(imgA)
    featsB = base_model(imgB)
   
    distance = tf.keras.layers.Lambda(euclidean_distance)([featsA, featsB])
    model = tf.keras.Model(inputs=[imgA, imgB], outputs=distance)
    model.compile(loss=contrastive_loss, optimizer=Adam(LR))
    return model

def get_image(label, val=False):
    """Choose an image from our training or val data with the
    given label."""
    if val:
        y = y_val; X = x_val
    else:
        y = y_train; X = x_train
    idx = np.random.randint(len(y))
    while y[idx] != label:
        # keep searching randomly!
        idx = np.random.randint(len(y))
    return X[idx]


def get_triplet(val=False):
    """Choose a triplet (anchor, positive, negative) of images
    such that anchor and positive have the same label and
    anchor and negative have different labels."""
    n = a = np.random.randint(num_classes)
    while n == a:
        # keep searching randomly!
        n = np.random.randint(num_classes)
    a, p = get_image(a, val), get_image(a, val)
    n = get_image(n, val)
    return a, p, n


def generate_triplets(val=False):
    """Generate an un-ending stream (ie a generator) of triplets for
    training or val."""
    while True:
        list_a = []
        list_p = []
        list_n = []

        for i in range(batch_size):
            a, p, n = get_triplet(val)
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)
            
        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')
        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(batch_size)
        yield (A, P, N), label

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def embedding_model_tl(inputShape, b_model, embeddingDim=128):
    
    base_model = get_base_model(b_model, inputShape)

    # freeze all the layers of EfficientNet, so they won't be trained.
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = tf.keras.layers.Input(shape=inputShape)
  
    x = base_model(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=embeddingDim)(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def complete_model_tl(base_model, LR):
    # Create the complete model with three
    # embedding models and minimize the loss 
    # between their output embeddings
    input_1 = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
    input_2 = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
    input_3 = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
        
    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
   
    loss = tf.keras.layers.Lambda(triplet_loss)([A, P, N]) 
    model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=loss)
    #model.compile(loss=identity_loss, optimizer=Adam(0.0001))
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model

def evaluate_model(x_test, y_test, anchors, model):
    # model = 'cl' - contrastive loss
    # model = 'tl' - triplet loss
    pred = []
    for i in range(len(y_test)):
        dists = []
        imgA = x_test[i].reshape(1, IMG_SIZE, IMG_SIZE, 3)
        if model == 'cl':
            for j in range(len(anchors)):
                imgB = anchors[j].reshape(1, IMG_SIZE, IMG_SIZE, 3)
                dist = model_cl.predict([imgA, imgB])[0][0]
                dists.append(dist)
            dists = np.array(dists)
            idx = np.argmin(dists)
            pred.append(idx)
        elif model == 'tl':
            featsA = base_model_tl(x_test[i].reshape(1, IMG_SIZE, IMG_SIZE, 3))
            for j in range(len(anchors)):
                imgB = anchors[j].reshape(1, IMG_SIZE, IMG_SIZE, 3)
                featsB = base_model_tl(anchors[j].reshape(1, IMG_SIZE, IMG_SIZE, 3))
                dist = np.linalg.norm(featsA - featsB, 2)
                dists.append(dist)
            dists = np.array(dists)
            idx = np.argmin(dists)
            pred.append(idx)
    pred=np.array(pred)
    accuracy = np.sum(y_test == pred) / len(y_test)
    precision = precision_score(y_test, pred, average='weighted')  # Use 'macro', 'micro', or 'weighted' as needed
    recall = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    return accuracy, precision, recall, f1


#######################################################################

#selection = [10, 13, 16, 17, 19, 20, 21, 24, 26, 30, 31, 34, 36, 44, 46, 49, 50, 51, 57, 59, 60, 61, 63, 65, 67, 68]
selection = [10, 13, 17, 19, 30, 31, 34, 61, 65, 67]

labels = [str(num).zfill(3) for num in selection] # aux list to format cat ids.

dir = 'hsc_dataset/' # directory of storage of cat images

csv_file = 'model_evaluation_results.csv' # CSV results path
results = []

if os.path.exists(csv_file):   # If CSV exists, load current results
    df_existing = pd.read_csv(csv_file)
    results = df_existing.to_dict('records')  # Converte o DataFrame para lista de dicionários
else:
    df_existing = pd.DataFrame()  # Cria um DataFrame vazio

## All hyperparameters combinations ##
photo_types = ['top','front']
base_models = ['vgg']
loss_function = ['triplet']
num_epochs = [100]
learning_rates = [0.0001]
augmentation = ['none','flip','noise','rotate']

hyperparameters = list(product(photo_types, base_models, loss_function, num_epochs, learning_rates, augmentation))

for hp in hyperparameters:
    photo_type, b_model, loss, epochs, learning_rate, aug = hp 

    if not df_existing.empty and ((df_existing['photo_type'] == photo_type)       & 
                                  (df_existing['base_model'] == b_model)          & 
                                  (df_existing['epochs'] == epochs)               &
                                  (df_existing['learning_rate'] == learning_rate) &
                                  (df_existing['loss_function'] == loss)          &
                                  (df_existing['augmentation'] == aug)).any():
        print(f"Combinação já avaliada: {hp}, pulando para a próxima...")
        continue

    try:
    
        ## LOADING IMAGES ##

        X = [] # initial an empty list X to store image of np.array()
        
        Z = [] # initial an empty list Z to store labels/names of cat individuals

        make_train_data(photo_type, labels) 

        le=LabelEncoder()
        Y=le.fit_transform(Z)
        
        X=np.array(X) # Transform and normalize X in the range of [0, 1]
        X=X/255.

        X.shape # shape of X, (number_image, image_size, image_size, channel_size)

        ## separate data train:val:test = 80:10:10 ##
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, stratify = Y, random_state=42)
        x_val,x_test,y_val,y_test=train_test_split(x_test,y_test, stratify = y_test, test_size=0.5,random_state=42)
        
        # Apply augmentation to training data
        if aug != 'none':
            
            augmentation_pipeline = get_augmentation_pipeline(aug)
            aug_X = []
            aug_Y = []

            for img, label in zip(x_train, y_train):
                img = img.astype(np.uint8)
                aug_img = augmentation_pipeline(image=img)  # Apply augmentation
                aug_X.append(aug_img)
                aug_Y.append(label)
            print(f"Original training size: {len(x_train)}")
            
            # Append augmented data to the training set
            x_train = np.concatenate([x_train, np.array(aug_X)])
            y_train = np.concatenate([y_train, np.array(aug_Y)])

            print(f"Augmented training size: {len(x_train)}")


        ## Generate pos/neg pairs for train_set, val_set ##
        (pairTrain, labelTrain) = create_pairs(x_train, y_train)
        (pairVal, labelVal) = create_pairs(x_val, y_val)

        ## CONTRASTIVE LOSS ##
        if loss == 'contrastive' or loss == 'all':
            base_model_cl = embedding_model_cl((IMG_SIZE, IMG_SIZE, 3), b_model)
            model_cl = complete_model_cl(base_model_cl, learning_rate)
            model_cl.summary()    
    
            history_cl = model_cl.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:], 
                        validation_data = ([pairVal[:, 0], pairVal[:, 1]], labelVal[:]),
                        batch_size = 16, epochs = epochs)
            
            anchor_images = [x_train[y_train==i][0] for i in range(num_classes)]
            anchor_images = np.array(anchor_images)
    
            acc_cl, precision_cl, recall_cl, f1_cl = evaluate_model(x_test, y_test, anchor_images, model = 'cl')
            print('The accuracy of contrastive loss on test set is {}%'.format(round(acc_cl * 100)))
            print('Precision of contrastive loss on the test set is {:.2f}'.format(precision_cl))
            print('Recall of contrastive loss on the test set is {:.2f}'.format(recall_cl))
            print('F1 Score of contrastive loss on the test set is {:.2f}'.format(f1_cl))
    
            results.append({
                'photo_type': photo_type,
                'base_model': b_model,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'loss_function':'constrastive',
                'accuracy': acc_cl,
                'precision': precision_cl,
                'recall': recall_cl,
                'f1_score': f1_cl,
                'augmentation': aug
            })
        
        ## TRIPLET LOSS ##
        if loss == 'triplet' or loss == 'all':
            output_signature = (
                (tf.TensorSpec(shape=(batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), 
                tf.TensorSpec(shape=(batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), 
                tf.TensorSpec(shape=(batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)),
                tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
            )
    
            train_generator = tf.data.Dataset.from_generator(
                generate_triplets,
                output_signature=output_signature
            )
    
            val_generator = tf.data.Dataset.from_generator(
                lambda: generate_triplets(val=True),
                output_signature=output_signature
            )
    
            base_model_tl = embedding_model_tl((IMG_SIZE, IMG_SIZE, 3), b_model)
            model_tl = complete_model_tl(base_model_tl, learning_rate)
            model_tl.summary()
    
            history_tl = model_tl.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                verbose=1,
                steps_per_epoch=20,
                validation_steps=30
            )

            anchor_images = [x_train[y_train==i][0] for i in range(num_classes)]
            anchor_images = np.array(anchor_images)
    
            acc_tl, precision_tl, recall_tl, f1_tl = evaluate_model(x_test, y_test, anchor_images, model = 'tl')
            print('The accuracy of triplet loss on test set is {}%'.format(round(acc_tl * 100)))
            print('Precision of triplet loss on the test set is {:.2f}'.format(precision_tl))
            print('Recall of triplet loss on the test set is {:.2f}'.format(recall_tl))
            print('F1 Score of triplet loss on the test set is {:.2f}'.format(f1_tl))
    
            results.append({
                'photo_type': photo_type,
                'base_model': b_model,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'loss_function':'triplet',
                'accuracy': acc_tl,
                'precision': precision_tl,
                'recall': recall_tl,
                'f1_score': f1_tl,
                'augmentation': aug
            })

        df = pd.DataFrame(results) #save partial results
        df.to_csv(csv_file, index=False)

        print(f"Resultados da combinação {hp} salvos com sucesso.")

    except Exception as e:
        print(f"Erro ao treinar ou avaliar o modelo com a combinação {hp}: {e}")
        continue  # Pula para a próxima combinação se der erro