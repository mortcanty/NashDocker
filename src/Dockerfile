FROM     debian:jessie
MAINTAINER Mort Canty "mort.canty@gmail.com"
ENV     REFRESHED_AT 2015-01-25
RUN apt-get update && apt-get install -y \
    python \
    build-essential \
    make \
    gcc \
    pandoc \
    lrslib \
    dos2unix \
    python-dev \
    python-pygments \
    python-numpy \
    python-pip 
    
RUN apt-get -y install python-cvxopt
    
RUN apt-get -y --no-install-recommends install python-sympy

RUN     pip install "ipython[notebook]"
ADD     notebook.sh /
RUN     chmod u+x /notebook.sh
RUN     dos2unix /notebook.sh

EXPOSE 8888

ENV     SCRIPTS_CHANGED 2015-02-03a
COPY    dist/ /dist
WORKDIR /dist
RUN     tar -xzvf nash-1.0.tar.gz
WORKDIR /dist/nash-1.0
RUN     python setup.py install 
COPY    tutorial.ipynb /home/tutorial.ipynb 
WORKDIR /home
CMD     ["/notebook.sh"]
