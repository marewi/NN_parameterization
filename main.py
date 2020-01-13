import getopt
import operator
import os
import pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
import xlrd
from termcolor import colored

from dataInput import data_input
from lib.generatePlot import write_event
from lib.toStringExt import paremetersToString, qtableToString, sheetToString
from modelTable import Model_table
from parameters import *
from qLearning import q_learning


def main(argv):
    # logs
    print(tf.__version__)
    # para = paremetersToString()
    # print(para)

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

    ### safe qtable results
    timestamp = time.time()
    with open(f"./logs/qtable_{data_filter}_{timestamp}.pkl", 'wb') as f:
        pickle.dump([q_table], f)
    print(f"SAVED: qtable_{data_filter}_{timestamp}.pkl")

    pt2 = qtableToString(q_table, type="directions")
    timestamp = time.time()
    qtable_directions = open(f"./logs/qtable_directions_{data_filter}_{timestamp}.txt", "w")
    qtable_directions.write(str(pt2))
    qtable_directions.close()
    print(f"SAVED: qtable_directions_{data_filter}_{timestamp}.txt")

    pt = qtableToString(q_table, type="values")
    timestamp = time.time()
    qtable_directions = open(f"./logs/qtable_values_{data_filter}_{timestamp}.txt", "w")
    qtable_directions.write(str(pt))
    qtable_directions.close()
    print(f"SAVED: qtable_values_{data_filter}_{timestamp}.txt")

    x, y = max(q_table.items(), key=operator.itemgetter(1))[0]
    print(colored(f"state with highest Q value: x={x}, y={y}", 'green'))

if __name__ == "__main__":
    main(sys.argv[1:])
