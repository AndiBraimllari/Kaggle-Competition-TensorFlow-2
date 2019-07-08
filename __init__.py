from image_preprocess import bootstrap_data_preprocess
from custom_conv_model import ConvModel
from model_training import train_animals

# Data sets are retrieved from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

if __name__ == '__main__':
    (train_data, test_data) = bootstrap_data_preprocess()
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope() as strat:
    model = ConvModel()
    train_animals(model, train_data, test_data)
