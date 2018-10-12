# NashDocker

The Docker container mort/nashdocker makes available an IPython wrapper which replaces some of the Mathematica functions in my  monograph on algorithmic game theory: Resolving Conflicts with Mathematica, Algorithms for Two-Person Games.

To run it on Ubuntu Linux for example, assuming you have Docker installed, simply type

sudo docker run -d -p 433:8888 --name=nash mort/nashdocker

and point your browser to localhost:433	

