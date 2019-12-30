import getopt
import operator
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
import xlrd
from termcolor import colored

from data_input import data_input
from environment import Goal, Square
from lib.generateplot import write_event
from lib.toStringExt import paremetersToString, sheetToString, qtableDirectionsToString
from model_table import Model_table
from parameters import *
from q_learning import q_learning


def main(argv):
    # logs
    print(tf.__version__)
    para = paremetersToString()
    print(para)

    ######################################
    # data filter
    data_filter = ''
    try:
        opts, args = getopt.getopt(argv,"s")
        print(opts, args)
    except getopt.GetoptError:
        print(colored("option doesnt exist", 'red'))
        sys.exit(2)
    if opts == [('-s', '')] and args == []: 
        raise Exception(colored("options for strategy was set, but no arguments", 'red'))
    if opts == []:
        print(colored("no data_filter was set", 'red'))
    else:
        for opt, _ in opts:
            print(f"opt: {opt}")
            if opt == '-s':
                print(f"param was {opt}")
                print(f"value was {args[0]}")
                data_filter = args[0]
                print(colored(f"data_filter set to: {data_filter}", 'green'))

    ######################################
    # backup old logs
    newpath = r'./oldlogs' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    source = './logs/'
    dest = './oldlogs/'
    files = os.listdir(source)
    for f in files:
        shutil.move(source+f, dest)

    ######################################
    # create environment
    env_time = time.time()
    loc = "./data/table_3000_clustered.xlsx"
    goals = data_input(loc, data_filter)

    print(colored(f"--- time to create environment: {time.time()-env_time} ---", 'blue'))

    ######################################
    # build model
    model_time = time.time()
    model = Model_table()
    print(colored(f"--- time to create q-table model: {time.time()-model_time} ---", 'blue'))

    ######################################
    # train model
    learning_time = time.time()
    episode_rewards, q_table, epsilons = q_learning(goals, model)
    print(colored(f"--- time to train model: {time.time()-learning_time} ---", 'blue'))

    ######################################
    # measure the results
    accumulated_reward = np.add.accumulate(episode_rewards)
    # moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
    # print(f"moving_avg: {moving_avg}")
    # print(f"q_table: {q_table}")

    ######################################
    # show results
    # TODO: dont save events in this dir because of git... this isnt so bad for logging for example
    # write_event(moving_avg, "moving_avg")
    # write_event(epsilons, "epsilon")
    plt1 = write_event(episode_rewards, 'rewards per episode')
    plt2 = write_event(accumulated_reward, 'accumulated reward')
    
    with tf.Session() as sess:
        summary = sess.run(plt1)
        summary2 = sess.run(plt2)
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary)
        writer.add_summary(summary2)
        writer.close()
    sess.close()

    # print qtable results
    pt = qtableDirectionsToString(q_table)
    qtable_directions = open("./logs/qtable_directions.txt", "w")
    qtable_directions.write(str(pt))
    qtable_directions.close()

if __name__ == "__main__":
    main(sys.argv[1:])
