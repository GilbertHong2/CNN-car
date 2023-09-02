# Here begins the code for the neural network model using tensorflow

## Load packages

from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model,Sequential
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# import argparse
from time import time

from skimage import exposure, color

from keras import backend as K
K.common.image_dim_ordering()
K.set_image_dim_ordering('tf')

## Initalize model
# The architectures used here are pretrained, such as resnet50/vgg19/inception.
# Other architectures can work as well.

def init_model(train_dir, val_dir, batch_size=32, model_name='resnet50', num_class=196, img_size=224):
    """
    initialize a cnn model and a training and validation data generator
    parms:
        args: parsed commandline arguments
    return:
        model: initialized model
        train_generator: training data generator
        validation_generator: validation data generator
    """

    print('loading the pre trained models...')

    # load base model
    if model_name == 'vgg19':
        base_model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape = (img_size, img_size, 3))
        preprocess_input = vgg19.preprocess_input
    elif model_name == 'vgg16':
        base_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape = (img_size, img_size, 3))
        preprocess_input = vgg16.preprocess_input
    elif model_name == 'inception_v3':
        base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape = (img_size, img_size,3))
        preprocess_input = inception_v3.preprocess_input
    elif model_name == 'resnet50':
        base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape = (img_size, img_size,3))
        preprocess_input = resnet50.preprocess_input

    # initalize training image data generator
    # hyperparameters here are to be fine tuned
    train_datagen = image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        # rescale=1./255,
        preprocessing_function=preprocess_input,
        # rotation_range=30,
        # shear_range=0.1,
        # zoom_range=0.1,
        vertical_flip=True,
        horizontal_flip=True
        )

    # initalize validation image data generator
    # hyperparameters here are also to be fine tuned
    validation_datagen = image.ImageDataGenerator(
        samplewise_center=True,
        # samplewise_std_normalization=True,
        rescale=1./255,
        preprocessing_function=preprocess_input
        )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')     

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        # color_mode='grayscale',  # 'rgb'
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

    # freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # add some customized layers
    x = base_model.output
    if model_name == 'vgg19':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        # x = Flatten(name='flatten')(x)
        # x = Dense(512, activation='relu', name='fc1-pretrain')(x)
        x = Dense(256, activation='relu', name='fc2-pretrain')(x)
        x = Dropout(0.3, name='dropout')(x)
    elif model_name == 'inception_v3':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(256, activation='relu', name='fc1-pretrain')(x)
        x = Dropout(0.3, name='dropout')(x)
    elif model_name == 'resnet50':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(256, activation='relu', name='fc1-pretrain')(x)
        x = Dropout(0.3, name='dropout')(x)

    # add softmax layer
    predictions = Dense(num_class, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam()
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    return model, train_generator, validation_generator


## Train model
# This part defines the function to train the model. 
# The hyperparameters can be modified.

def train(model, train_generator, validation_generator, num_class=196, model_name='resnet50', batch_size=32, epochs=30, suffix='laioffer'):
    """
    train the model
    parms:
        model: initialized model
        train_generator: training data generator
        validation_generator: validation data generator
        args: parsed command line arguments
    return:
    """
    # define number of steps/iterators per epoch
    stepsPerEpoch = train_generator.samples / batch_size
    validationSteps= validation_generator.samples / batch_size

    # save the snapshot of the model to local drive
    pretrain_model_name = 'pretrained_{}_{}_{}_{}.h5'.format(model_name, num_class, epochs, suffix)
    # visualize the training process
    tensorboard = TensorBoard(log_dir="logs/{}_pretrain_{}".format(model_name, time()), histogram_freq=0, write_graph=True)
    checkpoint = ModelCheckpoint(pretrain_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    earlystopping = EarlyStopping(monitor='acc', patience=5)
    callbacks_list = [checkpoint, tensorboard, earlystopping]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps)
    return history

