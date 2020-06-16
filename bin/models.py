import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter

class CNNModel1(nn.Module):
    def __init__(self, fully_layer_1, fully_layer_2, drop_rate):
        super(CNNModel1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 2)
        self.bn5 = nn.BatchNorm2d(32)

        """
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 64, 5)
        self.conv5 = nn.Conv2d(64, 32, 5)
        
        """
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_rate = drop_rate
        self.fc1 = nn.Linear(32*5*5, fully_layer_1)
        self.fc2 = nn.Linear(fully_layer_1, fully_layer_2)
        self.fc3 = nn.Linear(fully_layer_2, 2)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # print(x.shape)

        x = x.view(-1, 32*5*5)
        x = F.dropout(F.relu(self.fc1(x)), self.drop_rate)
        x = F.dropout(F.relu(self.fc2(x)), self.drop_rate)
        x = self.fc3(x)

        return x
"""

tflearn.layers.conv.conv_2d (incoming, nb_filter, filter_size, strides=1, padding='same', activation='linear', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D')
def CNNModel(outnode, model_name,  target, opt, learn_r, epch, n_of_h1, dropout_keep_rate, save_model=False):
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, outnode, activation='softmax')
    convnet = regression(convnet, optimizer=opt, learning_rate=learn_r, loss='categorical_crossentropy', name='targets')

    str_model_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(model_name,  target, opt, learn_r, epch, n_of_h1, dropout_keep_rate, save_model)

    model = None

    if save_model:
        model = tflearn.DNN(convnet, checkpoint_path='../tflearnModels/{}'.format(str_model_name), best_checkpoint_path='../tflearnModels/bestModels/best_{}'.format(str_model_name),
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="../tflearnLogs/{}/".format(str_model_name))
    else:
        model = tflearn.DNN(convnet)

    return model
"""