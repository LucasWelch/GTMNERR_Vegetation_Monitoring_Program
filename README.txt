Automation of the Vegetation Monitoring Program the GTMNERR Using Deep Neural Networks

Introduction:
Due to the importance of vegetation surveys to monitoring the health of environments, we have assisted Guana Tolomato Matanzas National Estuarine Research Reserve (GTMNERR) in automating the proccess using convolutional neural networks (CNNs) created using Keras and the Tensorflow framework. There are two different types of classifiers being used here--binary classifiers distinguishing between vegetated and unvegetated and a species classifier that distinguishes between the five species present in the marsh (spartina, juncus, batis, sarcocornia, and aviccenia). Contained in this repo are the 

What is needed: (latest version for each)
-Latest version of Python (3.9.0).
-Numpy
-Tensorflow
-Tensorflow GPU
-Keras

What is in this Repo:
-Python files for training binary and species classifier models.
-.SavedModel files for binary and species classifiers.

What is NOT in this Repo: These will be attainable via a link that is will be posted in the future.
-Data used for training.
-Images to be used by the final framework.

Using these files:
-Python files can be used to generate 10-fold cross validations for models.
	-After pulling files from Github and installing the above dependencies, change file paths for image_directory and master_file_path to match with the locations of the snippets and master CSV file.
	-Models generated will be saved in the same directory as the python files--10 SavedModel files will be generated for each program.

-SavedModel files can be used as models in our framework to make predictions (vegetated and unvegetated or a specific species).
	-With Keras, Tensorflow, and Numpy downloaded, call the keras.models.load_model('<path to model>').
	-This will return the saved model, which can then be used to make predictions.
	-Models have an input of size 31x31x3(rgb).


Model specifics:
The models output a vector with confidence scores (what the model believes the odds are the example is of a given class. Use the argmax function to return the index with the highest confidence (listed below for each type of model).
Binary:
	-Two classes (vegetated (0) and unvegetated(1))
Species:
	-Five classes (spartina (0), juncus (1), batis (2), sarcocornia (3), aviccenia (4))