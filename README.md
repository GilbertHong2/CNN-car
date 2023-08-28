# CNN-car
A Convolutional Neural Network for Car Classification

This project uses the Stanford car dataset attmpting to train a deep learning model to classify cars. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into training samples of size 8,144 and testing samples of size 8,041. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

The model I use will start with transfer learning to train the model. All neural network layers are fine tuned, and the last fully connected layer is entirely replaced.

Dataset (196 classes):

Train folder: 8144 images, avg: 41.5 images per class.

Test folder: 8041 images, avg: 41.0 images per class.

The dataset is available at https://ai.stanford.edu/~jkrause/cars/car_dataset.html.
