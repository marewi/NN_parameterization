# Optimization of a Neural Network by finding the optimal combination of parameters
using Q-Learning (https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)

This prototyp was developed as an container-based application to minimize the influence of the application to the execution environment and its operating system. Use the following commands to execute the docker-based application:

### build image
    docker build --rm -f Dockerfile -t nn_parameterization .

### run container
    docker run --rm -it -p 0.0.0.0:6006:6006 nn_parameterization

### conncecting to running container
    docker attach <container>

### run script
    python main.py
    
An alternative to copy the local project to the docker container, is so clone this repo into your running container:    
### clone repo in container
    git checkout <branch-name>
    git pull

### TODOS
Here are some completed tasks and some more ideas so optimize the performance of the RL Algorithm:
- [X] dont calculate new_q when running against barriers (e.g. penalizing when barrier)
- [ ] make Q-values independend on amount of visits of state-action pair
    - [X] random start
- [X] multithreading/-processing to catch errors in NN-training
- [X] instead of train NN in every state, use old reward if state was visited before
- [X] suggestion based on minimal loss instead of max V value
- [ ] multi-agent RL
- [ ] num_workers?
- [?] momentum instead of batch size
- [X] when is nan? --> loss is the large for datatype float
- [ ] partitioning of training process
    - [ ] save Q-values in table
- [ ] outsourcing of dynamic start generation in lib
