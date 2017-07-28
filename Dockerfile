FROM tensorflow/tensorflow:1.1.0

MAINTAINER Jonathan Langens <flowofcontrol@mail.com>

ADD *.py /scripts/

ADD *.sh /scripts/

ENV CLASSIFIER_TRESHHOLD 0.8

ENV FLASK_APP /scripts/classify.py

EXPOSE 6006

EXPOSE 5000

WORKDIR /tf_files

RUN pip install flask

RUN ln -s /scripts/retrain.sh /bin/retrain

RUN ln -s /scripts/start-server.sh /bin/start-server

RUN tensorboard --logdir training_summaries &

CMD start-server
