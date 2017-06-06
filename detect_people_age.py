import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from tensorflow.contrib.layers import *

class AgeClassifier(object):
    """
    use age:
    ac = AgeClassifier()
    
    image = cv2.imread('/root/dl-data/github/age_detect.bak/datasets/QQ20170515-151030.png')
    image2 = cv2.imread('/root/dl-data/github/age_detect.bak/datasets/QQ20170515-151116.png')
    image3 = cv2.imread('/root/dl-data/github/age_detect.bak/datasets/QQ20170515-151134.png')

    b = []
    b.append(image)
    b.append(image2)
    b.append(image3)
    
    results = ac.age_classify(b)
    """
    def __init__(self, checkpoint_path = 'models/inception/22801/checkpoint-14999'):
        """
        load model
        """
        self.label_list = ['(0, 2)','(3, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        self.nlabels = len(self.label_list)
        self.model_fn = self._inception_v3
        
        self.classify_res = []
        
        self.g = tf.Graph()
        with self.g.as_default():
            
            gpu_fraction=0.05
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
            self.sess = tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
                                                         log_device_placement=False))

            self.input1 = tf.placeholder(tf.float32, [None, 227, 227, 3])

            logits = self.model_fn(self.nlabels, self.input1, 1, False)
            self.softmax_output = tf.nn.softmax(logits)
               
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, checkpoint_path)
    
    def _inception_v3(self, nlabels, images, pkeep, is_training):
        batch_norm_params = {
            "is_training": is_training,
            "trainable": True,
            # Decay for the moving averages.
            "decay": 0.9997,
            # Epsilon to prevent 0s in variance.
            "epsilon": 0.001,
            # Collection containing the moving mean and moving variance.
            "variables_collections": {
                "beta": None,
                "gamma": None,
                "moving_mean": ["moving_vars"],
                "moving_variance": ["moving_vars"],
            }
        }
        weight_decay = 0.00004
        stddev=0.1
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        with tf.variable_scope("InceptionV3", "InceptionV3", [images]) as scope:

            with tf.contrib.slim.arg_scope(
                    [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    trainable=True):
                with tf.contrib.slim.arg_scope(
                        [tf.contrib.slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=batch_norm,
                        normalizer_params=batch_norm_params):
                    net, end_points = inception_v3_base(images, scope=scope)
                    with tf.variable_scope("logits"):
                        shape = net.get_shape()
                        net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                        net = tf.nn.dropout(net, pkeep, name='droplast')
                        net = flatten(net, scope="flatten")

        with tf.variable_scope('output') as scope:

            weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
            output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        return output
    
    def age_classify(self, image_list):
        """detect age
        Args:
            image_list : List with 3-D nparray front human face
        Return:
            Return:
            classify_res : list with every image in image_list : [lower,upper,probability]
            
            P.S.
            {
                lower : np.int32 the lower bound of the age
                upper : np.int32 the upper bount of the age
                probability : np.float32 the probability of the results
            }
	"""
        classify_res = []
        with self.g.as_default():
            crops = []
            
            for image_id in xrange(len(image_list)):
                crop = tf.image.resize_images(image_list[image_id], (227, 227))
                handled_image = tf.image.per_image_standardization(crop)
                crops.append(handled_image)

            image_batch = tf.stack(crops)
            image_batch = image_batch.eval(session=self.sess)
            batch_results = self.sess.run(self.softmax_output,feed_dict={self.input1:image_batch})
            
            for batch_result_id in xrange(len(batch_results)):
                output = batch_results[batch_result_id]
                batch_sz = batch_results.shape[0]
                
                best = np.argmax(output)
                best_choice = (self.label_list[best], output[best])
                lower = np.int32(best_choice[0][1:-1].split(',')[0])
                upper = np.int32(best_choice[0][1:-1].split(',')[1])
                probability = best_choice[1]
                classify_res.append([lower, upper, probability])
        return classify_res
