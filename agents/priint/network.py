''' import tensorflow as tf ! '''
import os

def Agent(data, opt, id_path):

    with open(id_path, 'w') as file:
        file.write("hello, I am an agent in training :) \nmy info is:")
        file.write("learning rate: " + str(opt.hyper.learning_rate))
        file.write("batch size: " + str(opt.hyper.batch_size))

