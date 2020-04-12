#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:34:57 2020

@author: rajan
"""


from blend_image import blend_image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import time
import string
import random
from PIL import Image
import numpy as np
import PIL.Image
from PIL import Image
import numpy as np
import PIL.Image


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}


def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model
    
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



def style_content_loss(outputs, style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image, extractor, style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers, opt):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs,style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))



def style_image(upload_foder, file_name, bgImage, stylefile):
    #UPLOAD_FOLDER = '/home/rajan/Documents/style-transfer/static'
    UPLOAD_FOLDER = os.getcwd()+'/static'
    print(file_name)
    print('in main -'+bgImage)
    bg_image_pw = UPLOAD_FOLDER+"/bg-images/"+bgImage
    
    bg_images = [bg_image_pw]

    fg_image = upload_foder+"/raw/"+file_name
    print('style_image:'+fg_image)
    blended_image = blend_image(fg_image, bg_images[0])

    #style_path = stylefile
    style_path = UPLOAD_FOLDER+"/style-images/"+stylefile


    content_image = load_img("Blended_image.jpg")
    style_image = load_img(style_path)


    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    img = tensor_to_image(stylized_image)
    

    img.save(UPLOAD_FOLDER+"/output_images/"+file_name[:-4]+"_"+bgImage[:-4]+"_"+stylefile[:-4]+".jpg")
    
    #return img
    """
    UPLOAD_FOLDER = '/home/dilsher/Downloads/style-transfer/static'
    print(file_name)
    print('in main -'+bgImage)
    bg_image_pw = UPLOAD_FOLDER+"/bg-images/"+bgImage
    
    bg_images = [bg_image_pw]

    fg_image = upload_foder+"/"+file_name
    
    blended_image = blend_image(fg_image, bg_images[0])

    #style_path = stylefile
    style_path = UPLOAD_FOLDER+"/style-images/"+stylefile


    content_image = load_img("Blended_image.jpg")
    style_image = load_img(style_path)

    
    
    x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')


    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2'] 

    # Style layer of interest
    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)    
    
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)
  
    extractor = StyleContentModel(style_layers, content_layers)

    results = extractor(tf.constant(content_image))

    style_results = results['style']
    
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']    
    
    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    style_weight=1e-2
    content_weight=1e4

    start = time.time()

    epochs = 2
    steps_per_epoch = 50
    
   step = 0
    for n in range(epochs):
      for m in range(steps_per_epoch):
        step += 1
        train_step(image, extractor, style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers, opt)
        print(".", end='')
      #display.clear_output(wait=True)
      img = tensor_to_image(image)
      print("Train step: {}".format(step))
      
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    
    
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    img = tensor_to_image(stylized_image)
    
    random_string = random_generator()
    img.save("/home/dilsher/Desktop/Capston-II/style-transfer/output_images/"+file_name[:-4]+random_string+"_styled.jpg")
    """
    #return img

    return file_name[:-4]+"_"+bgImage[:-4]+"_"+stylefile[:-4]+".jpg"



