# Optimization of a Neural Network by finding the optimal combination of parameters

### build image
    docker build --rm -f Dockerfile -t nn_parameterization .

### run container
    docker run --rm -it -p 0.0.0.0:6006:6006 nn_parameterization

### conncecting to running container
    docker attach <container>

### clone repo in container
    git checkout <branch-name>
    git pull

### run script
    python main.py

### TODOS
- [X] dont calculate new_q when running against barriers (e.g. penalizing when barrier)
- [ ] make Q-values independend on amount of visits of state-action pair
    - [X] random start
- [ ] partitioning of training process
    - [ ] save Q-values in table
- [ ] outsourcing of dynamic start generation in lib