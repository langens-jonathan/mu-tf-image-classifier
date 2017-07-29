import os, sys
import tensorflow as tf
import httplib2
import urllib
import json
import uuid

from flask import Flask
from subprocess import call

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

      json_string = "{\"label\": \"" + str(label) + "\", \"classifierTreshhold\":\"" + str(os.environ['CLASSIFIER_TRESHHOLD']) + "\", \"results\": ["

      isFirst = True

      for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if not isFirst:
          json_string += ", "
        else:
          isFirst = False
        json_string += "{\"label\":\"" + str(human_string) + "\", \"score\":\"" + str(score) + "\"}"

      json_string += "]}"
  
      return json_string

def get_filename_for_uuid(uuid):
    query = "SELECT ?file_name FROM <http://mu.semte.ch/application> WHERE { ?file <http://mu.semte.ch/vocabularies/core//uuid> \"" + uuid + "\" . ?file <http://mu.semte.ch/vocabularies/file-service/filename> ?file_name . }"
    resp, content = httplib2.Http().request("http://db:8890/sparql?query=" + urllib.quote_plus(query) + "&format=application%2Fsparql-results%2Bjson")

    json_result = json.loads(content)

    result_bindings = json_result['results']['bindings']

    if(len(result_bindings) < 1):
      return "FILE NOT FOUND"

    file_name = result_bindings[0]['file_name']['value']

    return file_name

def get_classname_for_uuid(uuid):
    query = "SELECT DISTINCT ?title FROM <http://mu.semte.ch/application> WHERE { ?s a <http://mu.semte.ch/vocabularies/ext/Class> . ?s <http://mu.semte.ch/vocabularies/core/uuid> \"" + uuid + "\" . ?s <http://purl.org/dc/terms/title> ?title . ?s ?p ?o . }"
    resp, content = httplib2.Http().request("http://db:8890/sparql?query=" + urllib.quote_plus(query) + "&format=application%2Fsparql-results%2Bjson")

    json_result = json.loads(content)

    result_bindings = json_result['results']['bindings']

    if(len(result_bindings) < 1):
      return "FILE NOT FOUND"

    classname = result_bindings[0]['title']['value']

    return classname

def update_file_location(uuid, old_location, new_location):
  query = "WITH <http://mu.semte.ch/application> DELETE { ?s <http://mu.semte.ch/vocabularies/file-service/filename> ?old_file_location . } INSERT { ?s <http://mu.semte.ch/vocabularies/file-service/filename> \"" + new_location + "\" . } WHERE { ?s ?p ?o . ?s <http://mu.semte.ch/vocabularies/core//uuid> ?uuid . ?s <http://mu.semte.ch/vocabularies/file-service/filename> ?old_file_location . FILTER(?old_file_location in (\"" + old_location + "\")) }"
  resp, content = httplib2.Http().request("http://db:8890/sparql?query=" + urllib.quote_plus(query) + "&format=application%2Fsparql-results%2Bjson")
  return ""

def insert_training_example_node_between(class_uuid, file_uuid):
  training_uuid = str(uuid.uuid4())
  training_uri = "http://example.com/image-classification/examples/" + training_uuid
  query = "WITH <http://mu.semte.ch/application> INSERT { <" + training_uri + "> a <http://mu.semte.ch/vocabularies/ext/TrainingExample> ; <http://mu.semte.ch/vocabularies/core/uuid> \"" + training_uuid + "\"; <http://mu.semte.ch/vocabularies/ext/hasFile> ?file. ?class <http://mu.semte.ch/vocabularies/ext/hasTrainingExample> <" + training_uri + "> . ?file <http://mu.semte.ch/vocabularies/core/uuid> ?file_uuid . } WHERE { ?file <http://mu.semte.ch/vocabularies/core//uuid> ?file_uuid . ?class <http://mu.semte.ch/vocabularies/core/uuid> ?class_uuid . FILTER(?file_uuid IN (\"" + file_uuid + "\")) FILTER(?class_uuid IN (\"" + class_uuid + "\")) }"
  resp, content = httplib2.Http().request("http://db:8890/sparql?query=" + urllib.quote_plus(query) + "&format=application%2Fsparql-results%2Bjson")
  return training_uuid
  
app = Flask(__name__)

@app.route('/classify/<file_uuid>')
def classify_route(file_uuid):
    filename = get_filename_for_uuid(file_uuid)
    return classify(filename)

@app.route('/retrain')
def retrain_route():
    call(["retrain"])
    return "retraining"

@app.route('/add-training-example/<class_uuid>/<file_uuid>')
def add_training_example(class_uuid, file_uuid):
  classname = get_classname_for_uuid(class_uuid)
  filename = get_filename_for_uuid(file_uuid)

  # ensuring the folder
  call(["mkdir",  "-p",  "/images/" + classname])

  # moving the file
  file_location = "/images/" + classname + "/" + file_uuid
  call(["mv", filename, file_location])

  # update the location in the db
  update_file_location(file_uuid, filename, file_location)

  # inserting the training example and links in the db
  training_example_uuid = insert_training_example_node_between(class_uuid, file_uuid)
  
  return "{\"trainingExampleUUID\":\"" + training_example_uuid + "\"}"

app.run(host= '0.0.0.0')
