import time
import argparse
import os
import sys
if sys.version_info >= (3, 0):
        import _pickle as cPickle
else:
        import cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS

##In the training of a SVM, we are trying to find the optimal 'Hyperplane' to seperate classes of data. A hyperplane is nothing more than a plane in a dimension lower than the total number of dimensions we are working in.
##For example, if we have three different features, then we work in a 3 dimensional space. Therefore, our hyperplane which seperates the classes is twodimensional: A plane.
##If there are only two different features, the hyperplane would be a line that seperates the classes as well as possible. To learn more about the basic workings of a SVM, please visit my 'cosde insight' document.
def train(epochs=HYPERPARAMS.epochs, random_state=HYPERPARAMS.random_state, ##Get all necessary hyperparameters, and train the model (if you load in an already trained model, train_model can be turned to False.)
          kernel=HYPERPARAMS.kernel, decision_function=HYPERPARAMS.decision_function, gamma=HYPERPARAMS.gamma, train_model=True):

        print( "loading dataset " + DATASET.name + "...")
        if train_model: #Train model
                data, validation = load_data(validation=True)
        else: #Or simply run the test data
                data, validation, test = load_data(validation=True, test=True)
        
        if train_model:
            # Training phase
            print( "building model...")
            model = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function, gamma=gamma) ##Uses the scikit learning library to create a SVM with the parameters as set
            ##in 'parameters.py'
            ##All these print statements just give the user a lot of info
            print( "start training...")
            print( "--")
            print( "kernel: {}".format(kernel))
            print( "decision function: {} ".format(decision_function))
            print( "max epochs: {} ".format(epochs))
            print( "gamma: {} ".format(gamma))
            print( "--")
            print( "Training samples: {}".format(len(data['Y'])))
            print( "Validation samples: {}".format(len(validation['Y'])))
            print( "--")
            ##Training time is already being measured!!
            start_time = time.time()
            model.fit(data['X'], data['Y'])
            training_time = time.time() - start_time
            print( "training time = {0:.1f} sec".format(training_time))

            if TRAINING.save_model: ##After training has been completed, one can decide to have the mdel saved.
                print( "saving model...")
                with open(TRAINING.save_model_path, 'wb') as f:
                        cPickle.dump(model, f)

            print( "evaluating...")
            validation_accuracy = evaluate(model, validation['X'], validation['Y']) ##Calculation the accuracy
            print( "  - validation accuracy = {0:.1f}".format(validation_accuracy*100))
            return validation_accuracy
        else:
            # Testing phase : load saved model and evaluate on test dataset
            print( "start evaluation...")
            print( "loading pretrained model...")
            if os.path.isfile(TRAINING.save_model_path):
                with open(TRAINING.save_model_path, 'rb') as f:
                        model = cPickle.load(f)
            else:
                print( "Error: file '{}' not found".format(TRAINING.save_model_path))
                exit()

            print( "--")
            print( "Validation samples: {}".format(len(validation['Y'])))
            print( "Test samples: {}".format(len(test['Y'])))
            print( "--")
            print( "evaluating...")
            ##Testing period is also timed :)
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'],  validation['Y'])
            print( "  - validation accuracy = {0:.1f}".format(validation_accuracy*100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print( "  - test accuracy = {0:.1f}".format(test_accuracy*100))
            print( "  - evalution time = {0:.1f} sec".format(time.time() - start_time))
            return test_accuracy

def evaluate(model, X, Y):
        predicted_Y = model.predict(X) #The predicated label based on the SVM's output
        accuracy = accuracy_score(Y, predicted_Y) #A comparison of the actual label and the SVM output, based on some accuracy score formula (standard metric imported from scikit learn library)
        return accuracy

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
parser.add_argument("-m", "--max_evals", help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
        train(train_model=False)