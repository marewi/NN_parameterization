# finding a point in a grid layout

### going into running container
    docker attach <container>

### build image
    docker build --rm -f Dockerfile -t nn_parameterization .

### run container
    docker run --rm -it -p 0.0.0.0:6006:6006 nn_parameterization

### clone repo in container
    git checkout <branch-name>
    git pull

### run script
    python main.py -s window/triangle/hook/0/1/2

### evaluate algorithm
    python evaluate.py -f logs/qtable_3_1578156572.3556674.pkl