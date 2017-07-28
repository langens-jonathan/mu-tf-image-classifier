import os, sys

import tensorflow as tf
from flask import Flask

tf_files = "/tf_files/"

def classify(image_path):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
  
  # Read in the image_data
  image_data = tf.gfile.FastGFile(image_path, 'rb').read()
  
  # Loads label file, strips off carriage return
  label_lines = [line.rstrip() for line 
                     in tf.gfile.GFile(tf_files + "retrained_labels.txt")]
  
  # Unpersists graph from file
  with tf.gfile.FastGFile(tf_files + "retrained_graph.pb", 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')
  
  with tf.Session() as sess:
      # Feed the image_data as input to the graph and get first prediction
      softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
      
      predictions = sess.run(softmax_tensor, \
               {'DecodeJpeg/contents:0': image_data})
      
      # Sort to show labels of first prediction in order of confidence
      top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
  
      for node_id in top_k:
          human_string = label_lines[node_id]
          score = predictions[0][node_id]
          print('%s (score = %.5f)' % (human_string, score))
  
      score = predictions[0][top_k[0]]
  
      label = label_lines[top_k[0]]
  
      print('top score(treshhold): %s(%s)' % (score, os.environ['CLASSIFIER_TRESHHOLD']))
  
      if(float(score) < float(os.environ['CLASSIFIER_TRESHHOLD'])):
          label = "unclassified"
  
      print('%s' % label)
      
      return label
    
# change this as you see fit
# image = sys.argv[1]

app = Flask(__name__)

@app.route('/')
def classify_route():
    return classify("/files/IMG_6501.jpg")

app.run(host= '0.0.0.0')
