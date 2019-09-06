''' import tensorflow as tf ! '''
import os

def Agent(data, opt, output_path):

    id_path = output_path + opt.output_path + '/checkpoint.txt'
    dirname = os.path.dirname(id_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(id_path, 'w') as file:
        file.write("hello, I am an agent in training :) \nmy info is:")
        file.write("learning rate: " + str(opt.hyper.learning_rate))
        file.write("batch size: " + str(opt.hyper.batch_size))

