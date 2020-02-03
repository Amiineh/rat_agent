import csv
import os
import re
import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
sns.set_style('ticks')
sns.set_context("poster")

main_path = '/Users/amineh.ahm/Desktop/Mice/code/rat_exp/acktr_dm_shaping'
name = 'rat1_2019-02-04_09_52_53.csv'
num_rats = 16

shape = 2
rat = 12

possible_events_s1 = ['in_base', 'reward_time_started', 'not_in_base', 'rat_left_base_during_reward_time',
                      'correct_trial', 'correct_or_late', 'left_early', 'already_in_base', 'missed_trial', 'error']

possible_events_s2 = ['in_base', 'distractor_time_started', 'distractor_avoided',
                      'not_in_base', 'left_early', 'rat_left_during_distractor', 'rat_left_base_during_reward_time',
                      'correct_trial', 'correct_or_late', 'missed_trial']


def csv_datestr_to_date_filename(datestr):
    """
    Does date conversions for the date in the filenames.
    """
    # ex = "2019-02-20_11_23_33"
    return datetime.datetime.strptime(datestr, '%Y-%m-%d_%H_%M_%S')


def csv_datestr_to_date_csvdata_2(datestr):
    """
    Does date conversions the date in the csvs used as timestamps, but cuts out the hours minutes seconds
    """
    # ex = "66_33:4:38"
    newdatestr = datestr.split('_')
    return int(newdatestr[0])
    # return datetime.datetime.strptime(newdatestr[0], '%d')


def csv_datestr_to_date_csvdata(datestr):
    """
    Does date conversions the date in the csvs used as timestamps.
    """
    # ex = "2019-02-06_11:18:36"
    return datetime.datetime.strptime(datestr, '%d_%H:%M:%S')


# getting data


def csv_to_dict_condensed(filepath, shape):
    """
    Given a filepath, convert the csv to a dictionary called data, with keys
    corresponding to column labes and values the list of all the columns.

    This also procceses the data to create a dictionary called sequence that stores
    the sequence of events with timestamps.
    """
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        header_row = next(spamreader)

        # Deal with multiple delimiter types
        if len(header_row) == 1:
            header_row = header_row[0].split(',')

        data = dict()
        for col_name in header_row:
            data[col_name] = []

        cur = [0, 0, 0, 0, 0, 0]
        sequence = dict()
        for row in spamreader:
            if len(row) == 0:
                # skip empty rows
                continue
            if row[0] == 'time_stamp':
                # skip first row
                continue

            vals = re.split(',| ', row[0])

            if (vals[1] == 'in_base' or vals[1] == 'already_in_base' or vals[1] == 'distractor_avoided'):
                new = [int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]
                if new[2] < 0:
                    new[2] = 0
                mark = -1
                for i in range(len(cur)):
                    if (new[i] - cur[i] > 0):
                        sequence[vals[0]] = i
                        mark = i
                        break
                for i in range(len(cur)):
                    cur[i] = new[i]

            if (vals[1] == 'missed_trial'):
                sequence[vals[0]] = 0
                cur[0] += 1

            for i, col_name in enumerate(header_row):
                data[col_name].append(vals[i])

        return sequence  # data still holds the entire thing, but sequence is just a list of events with times


def extract_return_time(filepath, shape):
    """
    Takes in the filepath and stage/shaping to return a list of the return and reaction times.
    """
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        header_row = next(spamreader)

        # Deal with multiple delimiter types
        if len(header_row) == 1:
            header_row = header_row[0].split(',')

        data = dict()
        for col_name in header_row:
            data[col_name] = []

        returntimes = []
        reactiontimes = []
        pastaction = ' '
        past2action = ' '
        pasttime = 0
        for row in spamreader:
            if len(row) == 0:
                # skip empty rows
                continue
            if row[0] == 'time_stamp':
                # skip first row
                continue

            vals = re.split(',| ', row[0])
            time = vals[0].split('_')[1]
            time = time.split(':')
            seconds = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
            timediff = seconds - pasttime
            if timediff < 0:
                break
            if (vals[1] == 'not_in_base' and pastaction == 'reward_time_started'):
                reactiontimes.append(timediff)
            if (vals[1] == 'in_base' and past2action == 'correct_trial'):
                returntimes.append(timediff)

            past2action = pastaction
            pastaction = vals[1]
            pasttime = seconds

        return reactiontimes, returntimes


def graph_mean_reaction_return(path, shape, train, yreaction, yreturn):
    """
    Uses the extract_return_time function to plot the mean and standard deviation of the
    data at the given path.

    For the AI, this is just the plot for one of the paths instead of averaging all beccuase
    the graphs get too messy.
    """
    returnavgs = {}
    returnstds = {}
    reactionavgs = {}
    reactionstds = {}

    for root, dirs, files in os.walk(path):
        for name in sorted(files):
            filepath = os.path.join(root, name)
            if ".csv" in name:
                rat = int(name.split('_')[0][3:])
                day = int(name.split('_')[1][:-4])
                reactiontimes, returntimes = extract_return_time(filepath, shape)
                if len(reactiontimes) == 0:
                    meanreaction = 0
                else:
                    meanreaction = sum(reactiontimes) / len(reactiontimes)
                if len(returntimes) == 0:
                    meanreturn = 0
                else:
                    meanreturn = sum(returntimes) / len(returntimes)

                if day not in returnavgs.keys():
                    returnavgs[day] = np.zeros(num_rats)
                    returnstds[day] = np.zeros(num_rats)
                    reactionavgs[day] = np.zeros(num_rats)
                    reactionstds[day] = np.zeros(num_rats)

                returnavgs[day][rat] = meanreturn
                returnstds[day][rat] = np.std(returntimes)
                reactionavgs[day][rat] = meanreaction
                reactionstds[day][rat] = np.std(reactiontimes)

    if train == False:
        phrase = "From scratch"
    else:
        phrase = "Shaping"
    fig, ax = plt.subplots(figsize=(8, 6))
    csfont = {'fontname': 'Times New Roman'}

    # graph reaction
    for i in range(num_rats):
        times = []
        avgs = []
        stds = []
        for j in sorted(returnavgs.keys()):
            if reactionavgs[j][i] != 0:
                times.append(j * 16)
                avgs.append(reactionavgs[j][i])
                stds.append(reactionstds[j][i])
        ax.errorbar(times, avgs, stds, linewidth=3)
        break
    # ax.legend(loc='best', fontsize = 15)
    ax.set_ylim(bottom=0)
    if yreaction != 0:
        ax.set_ylim(0, yreaction)
    ax.set_xlabel('Session Number', **csfont)
    ax.set_ylabel('Seconds', **csfont)
    ax.set_title('Mean Reaction Time for \n Stage ' + str(shape) + ' - ' + str(phrase), **csfont)
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)
    # plt.show()

    # plt.savefig("reaction" + str(train) + str(shape) + '.pdf', dpi=1000)
    plt.show()

    # graph return
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_rats):
        times = []
        avgs = []
        stds = []
        for j in sorted(returnavgs.keys()):
            if returnavgs[j][i] != 0 and returnavgs[j][i] < 200:
                times.append(j * 16)
                avgs.append(returnavgs[j][i])
                stds.append(returnstds[j][i])
        ax.errorbar(times, avgs, stds, linewidth=3)
        break
    # ax.legend(loc='best', fontsize = 15)
    ax.set_ylim(bottom=0)
    if yreturn != 0:
        ax.set_ylim(0, yreturn)
    ax.set_xlabel('Session Number', **csfont)
    ax.set_ylabel('Seconds', **csfont)
    ax.set_title('Mean Return Time for \n Stage ' + str(shape) + ' - ' + str(phrase), **csfont)
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)
    # plt.show()

    # plt.savefig("return" + str(train) + str(shape) + '.pdf', dpi=1000)


def extract_consecutive_data(path, shape):
    size = 6
    fullRecord = np.zeros(size)
    data = csv_to_dict_condensed(path, shape)
    # print(data)
    counts = np.zeros((size, size))
    order = []
    time = 0
    for i in data.keys():
        order.append(data[i])
        fullRecord[data[i]] += 1
        time = csv_datestr_to_date_csvdata_2(i)
    for i in range(1, len(order)):
        counts[order[i]][order[i - 1]] += 1
    return time, fullRecord, counts


def normalize(counts, size):
    """
    To test independence, looks at conditional probabilities
    """
    rowsum = np.zeros(size)
    total = 0
    for i in range(size):
        for j in range(size):
            rowsum[i] += counts[i][j]
        total += rowsum[i]

    twodcounts = []  # still contains unnormalized 2d counts, just isn't graphed right now
    conditional2d = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            twodcounts.append(counts[i][j])
            if counts[i][j] == 0:
                continue
            conditional2d[i][j] = counts[i][j] * 1.0 / total  # changes from numerical into probabilities
            conditional2d[i][j] = conditional2d[i][j] * 1.0 / (rowsum[i] / total)  # conditional probabilities
    return conditional2d


def compile_and_normalize(path, shape, rat):
    size = 6
    counts = np.zeros((size, size))  # counts consecutives
    fullRecord = np.zeros(size)  # total number of m, e, l, c
    segdict = {}  # stores by day, the number of m, e, l, c
    for root, dirs, files in os.walk(path):
        numfiles = len(files) - 1
        files.sort()
        count = 0
        for name in files:
            count += 1
            filepath = os.path.join(root, name)
            ratid = "rat" + str(rat)
            if ratid not in filepath:
                continue
            if ".csv" in name:
                time, fullRecord1, counts1 = extract_consecutive_data(filepath, shape)
                segdict[time] = fullRecord1
                # print(filepath, time, fullRecord1)
                for i in range(size):
                    fullRecord[i] += fullRecord1[i]
                    for j in range(size):
                        counts[i][j] += counts1[i][j]

    # print("segdict", segdict)
    # print("fullRecord", fullRecord)
    conditional2d = normalize(counts, size)

    return segdict, fullRecord, conditional2d


def graph_normalized_overtime(shape, train):
    size = 6

    norm = {}
    uncond = {}
    timecounts = {}
    totaldict = {}
    avg = {}
    std = {}
    # this one's not fixed for AI format yet
    # instead maybe i don't need to iterate through the rats, just count them
    path = main_path + '/train_' + str(rat) + '/reports/'
    print(path)
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = os.path.join(root, name)
            if ".csv" in name:
                newname = name.split("_")
                i = int(newname[0][3:])
                time, fullRecord, counts = extract_consecutive_data(filepath, shape)
                # print(i, filepath, time, fullRecord)
                fsum = sum(fullRecord)
                for j in range(size):
                    fullRecord[j] /= fsum
                conditional = normalize(counts, size)
                if time not in norm:
                    norm[time] = conditional
                    timecounts[time] = 1
                    uncond[time] = np.zeros((size, num_rats))
                    totaldict[time] = np.zeros((size, size, num_rats))
                    for j in range(size):
                        for k in range(size):
                            for l in range(num_rats):
                                totaldict[time][j][k][l] = -1
                                uncond[time][j][l] = -1
                else:
                    timecounts[time] += 1
                    for j in range(size):
                        for k in range(size):
                            norm[time][j][k] += conditional[j][k]
                for j in range(size):
                    for k in range(size):
                        totaldict[time][j][k][i] = conditional[j][k]
                    uncond[time][j][i] = fullRecord[j]

    # print(uncond)
    for key in norm.keys():
        for i in range(size):
            for j in range(size):
                norm[key][i][j] /= timecounts[key]

    # print("norm", norm)
    # print("uncond", uncond)

    print("to test independence:")
    colors = ["red", "orange", "black", "purple", "blue", "green"]
    labels = ['missed', 'early', 'late', 'distracted', 'distracted ok', 'correct']
    wanted = [0, 1, 2, 3, 4, 5]

    times = sorted(totaldict.keys())
    for i in range(len(times)):
        times[i] = times[i] * 16
    uncondavg = {}
    unconderr = {}
    for key in sorted(totaldict.keys()):
        uncondavg[key] = np.zeros(size)
        unconderr[key] = np.zeros(size)
        for i in range(size):
            row = []
            for j in range(num_rats):
                if (uncond[key][i][j] != -1):
                    row.append(uncond[key][i][j])
            uncondavg[key][i] = sum(row) / len(row)
            unconderr[key][i] = np.std(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(size):
        arravg = []
        arrstd = []
        for key in sorted(uncondavg.keys()):
            arravg.append(uncondavg[key][i])
            arrstd.append(unconderr[key][i])
        # plt.plot(times, arravg, marker = 'o', color = colors[i], label = labels[i])
        ax.errorbar(times, arravg, arrstd, linewidth=3, color=colors[i], label=labels[i])
    ax.legend(loc='best', fontsize=15)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Day count')
    ax.set_ylabel('Trial frequency')
    ax.set_title('Probabilities of all rats from shaping ' + str(shape) + ' for training ' + str(train))
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)
    plt.show()

    for key in sorted(totaldict.keys()):
        avg[key] = np.zeros((size, size))
        std[key] = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                row = []
                for k in range(num_rats):
                    if (totaldict[key][i][j][k] != -1):
                        row.append(totaldict[key][i][j][k])
                avg[key][i][j] = sum(row) / len(row)
                std[key][i][j] = np.std(row)

    for i in range(len(wanted)):
        fig, ax = plt.subplots(figsize=(8, 6))
        for j in range(size):
            arravg = []
            arrstd = []
            for key in sorted(avg.keys()):
                arravg.append(avg[key][wanted[i]][j])
                arrstd.append(std[key][wanted[i]][j])
            ax.errorbar(times, arravg, arrstd, linewidth=3, color=colors[j], label=labels[i])
        ax.set_title(
            'Conditional probabilities given ' + labels[wanted[i]] + ' trial with stddev of shaping ' + str(shape))
        ax.legend(loc='best', fontsize=15)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Day count')
        ax.set_ylabel('Trial frequency')
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)
        plt.show()

        split = [[], [], [], [], [], []]
        for key in sorted(norm.keys()):
            for j in range(size):
                split[j].append(norm[key][wanted[i]][j])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.stackplot(times, split[0], split[1], split[2], split[3], split[4], split[5], labels=labels)
        ax.legend(loc='best', fontsize=15)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Day count')
        ax.set_ylabel('Trial frequency')
        ax.set_title('Conditional probabilities given ' + labels[wanted[i]] + ' trial of shaping ' + str(shape))
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)
        plt.show()


def graph_segmented_days(data,
                         shape):  # data is dict with timestamps as keys and array of [m, e, l, c] as values for that day/time
    '''
    does sanity check so shows the progression of trials over all trials for 1 rat
    '''
    # print("data", data)
    times = []
    segmented = [[], [], [], [], [], []]
    size = 6

    for i in sorted(data.keys()):
        times.append(i * 16)
        for j in range(size):
            segmented[j].append(data[i][j])

    labels = ["correct", "distracted ok", "distracted", "early", "late", "missed"]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.stackplot(times, segmented[5], segmented[4], segmented[3], segmented[1], segmented[2], segmented[0],
                  labels=labels)
    ax.legend(loc='lower right', fontsize=15)
    # ax.set_ylim(0,1)
    ax.set_xlabel('Day count')
    ax.set_ylabel('Trial frequency')
    ax.set_title("Stacked frequencies")
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)

    '''if shape == 1:
        plt.ylim((0,200))
    else: 
        plt.ylim((0,120))'''

    plt.show()


def combined_rats_total(shape, gtype, legend, xlim, ylim):  # gtype is counts or frequencies
    size = 6
    sumdict = {}  # adds up for all days per trial type (this one isn't really needed)
    totaldict = {}  # keeps values to calculate std dev
    numcounts = {}  # keep track of how many rats had data for each day
    for i in range(num_rats):
        path1 = main_path
        segdict, fullRecord, counts = compile_and_normalize(path1, shape, i)
        # if i want just counts and not frequencies, comment out the next for loop
        if gtype == "frequencies":
            for key in segdict.keys():
                rsum = 0
                for j in range(size):
                    rsum += segdict[key][j]
                for j in range(size):
                    segdict[key][j] /= rsum
        for key in sorted(segdict.keys()):
            if key not in sumdict:
                numcounts[key] = 0
                sumdict[key] = np.zeros(size)
                totaldict[key] = np.zeros((size, num_rats))
                for a in range(size):
                    for b in range(num_rats):
                        totaldict[key][a][b] = -1
            numcounts[key] += 1
            for j in range(size):
                sumdict[key][j] += segdict[key][j]
                totaldict[key][j][i - 1] = segdict[key][j]

    for key in sorted(sumdict.keys()):
        for j in range(size):
            sumdict[key][j] /= numcounts[key]

    colors = ["red", "orange", "black", "purple", "blue", "green"]
    labels = ['missed', 'early', 'late', 'distracted', 'distracted ok', 'correct']
    avgs = [[], [], [], [], [], []]
    err = [[], [], [], [], [], []]

    for i in range(size):
        for key in sorted(totaldict.keys()):
            terms = []
            for j in range(len(totaldict[key][i])):
                if totaldict[key][i][j] != -1:
                    terms.append(totaldict[key][i][j])
            avgs[i].append(sum(terms) / len(terms))
            err[i].append(np.std(terms))

    x = sorted(totaldict.keys())
    for i in range(len(x)):
        x[i] = x[i] * 16

    fig, ax = plt.subplots(figsize=(8, 6))
    csfont = {'fontname': 'Times New Roman'}

    if shape == 1:
        want = [0, 1, 2, 5]
    else:
        want = [0, 1, 2, 3, 4, 5]

    for i in want:
        # plt.plot(x, avgs[i], marker = 'o', color = colors[i], label = labels[i])
        ax.errorbar(x, avgs[i], err[i], linewidth=3, color=colors[i], label=labels[i])
    if legend:
        ax.legend(loc='best', fontsize=15)
    if ylim != 0:
        ax.set_ylim(0, ylim)
    if xlim != 0:
        ax.set_xlim(0, xlim)
    ax.set_xlabel('Session Number', **csfont)
    ax.set_ylabel('Trial ' + gtype, **csfont)
    # if pretrain == False:
    #     phrase = "From scratch"
    # else:
    #     phrase = "Shaping"
    # ax.set_title('Stage ' + str(shape) + ' - ' + phrase, **csfont)
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.17)
    # plt.show()

    # plt.savefig(gtype + str(pretrain) + str(shape) + '.pdf', dpi=1000)

    graph_segmented_days(sumdict, shape)


def graph_segmented_perday(path, size):
    # graphs it over the 30 minutes to test if the rats ever get hungry
    # THIS ONE CAN'T BE DIRECTED USED FOR SIZE 6 SHAPING 2 SO AHVE TO CHANGE THAT

    size = 6
    totalpermin = np.zeros((size, 30))
    mins = []
    for i in range(30):
        mins.append(i)
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = os.path.join(root, name)
            if ".csv" in name:
                # filepath = ./csv/rat4_shaping2/rat4_2019-02-20_11_23_33.csv
                # name = rat4_2019-02-20_11_23_33.csv
                times = []
                vals = []
                split = [[], [], [], [], [], []]
                permin = [[], [], [], [], [], []]
                data = csv_to_dict_condensed(filepath, shape)
                for key in data.keys():
                    stime = key[11:16]
                    times.append(stime)
                    vals.append(data[key])
                start = times[0]
                cur = [int(start[:2]), int(start[3:])]
                end = times[len(times) - 1]
                done = [int(end[:2]), int(end[3:])]
                minutes = []
                while (cur != done):
                    minutes.append(str(cur[0]) + '.' + str(cur[1]))
                    cur[1] += 1
                    if cur[1] == 60:
                        cur[1] = 0
                        cur[0] += 1
                    for i in range(4):
                        permin[i].append(0)
                minutes.append(str(cur[0]) + '.' + str(cur[1]))
                for i in range(size):
                    permin[i].append(0)

                split = minutes[0].find('.')
                shour = int(minutes[0][:split])
                smin = int(minutes[0][split + 1:])
                for i in range(len(times)):
                    csplit = times[i].find(':')
                    chour = int(times[i][:csplit])
                    cmin = int(times[i][csplit + 1:])
                    posat = cmin - smin
                    if posat < 0:
                        posat = posat + 60
                    permin[vals[i]][posat] += 1

                for i in range(len(permin)):
                    minlen = min(len(totalpermin[0]), len(permin[0]))
                    for j in range(minlen):
                        totalpermin[i][j] += permin[i][j]

    print("mins", mins)
    print("totals", totalpermin)

    fig, ax = plt.subplots()
    labels = ["correct", "distracted ok", "distracted", "early", "late", "missed"]
    ax.stackplot(mins, totalpermin[5], totalpermin[4], totalpermin[3], totalpermin[1], totalpermin[2], totalpermin[0],
                 labels=labels)
    ax.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':

    # combined_rats_total(shape, "counts", False, 0, 150)
    graph_normalized_overtime(shape, rat)

    # path = main_path + '/train_' + str(rat) + '/reports/'
    # for i in [rat]:  # range(num_rats):
    #     print(path, i)
    #     segdict, fullRecord, counts = compile_and_normalize(path, shape, i)
    #     # print(segdict)
    #     print(fullRecord)
    #     graph_segmented_days(segdict, shape)
    #     break


    # for mean reaction return times, the format is
    # # graph_mean_reaction_return(path, shape, train, yreaction, yreturn)
    # print("training 4, no pretrain")
    # path0 = '/Users/alexandra_li/college/year1/urop/rat_data/train_4/reports'
    # # graph_mean_reaction_return(path0, 1, False, 8, 200)

    # print("training 4, pretrained")
    # path1 = '/Users/alexandra_li/college/year1/urop/rat_data/train_4_pretrain0/reports'
    # # graph_mean_reaction_return(path1, 1, True, 8, 200)

# # !/usr/bin/env python
# # coding: utf-8
#
#
# import csv
# import os
# import re
# import datetime
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats
# import pandas
# from scipy.stats import norm
# from functools import reduce
# from itertools import islice
#
# main_path = '/Users/amineh.ahm/Desktop/Mice/code/rat_exp/a2c_dm_scratch'
# name = 'rat1_2019-02-04_09_52_53.csv'
# num_rats = 16
#
# possible_events_s1 = ['in_base', 'reward_time_started', 'not_in_base', 'rat_left_base_during_reward_time',
#                       'correct_trial', 'correct_or_late', 'left_early', 'already_in_base', 'missed_trial', 'error']
#
# possible_events_s2 = ['in_base', 'distractor_time_started', 'distractor_avoided',
#                       'not_in_base', 'left_early', 'rat_left_during_distractor', 'rat_left_base_during_reward_time',
#                       'correct_trial', 'correct_or_late', 'missed_trial']
#
#
# def csv_datestr_to_date_filename(datestr):
#     """
#     Does date conversions for the date in the filenames.
#     """
#     # ex = "2019-02-20_11_23_33"
#     return datetime.datetime.strptime(datestr, '%Y-%m-%d_%H_%M_%S')
#
#
# def csv_datestr_to_date_csvdata_2(datestr):
#     """
#     Does date conversions the date in the csvs used as timestamps, but cuts out the hours minutes seconds
#     """
#     # ex = "66_33:4:38"
#     newdatestr = datestr.split('_')
#     return int(newdatestr[0])
#     # return datetime.datetime.strptime(newdatestr[0], '%d')
#
#
# def csv_datestr_to_date_csvdata(datestr):
#     """
#     Does date conversions the date in the csvs used as timestamps.
#     """
#     # ex = "2019-02-06_11:18:36"
#     return datetime.datetime.strptime(datestr, '%d_%H:%M:%S')
#
#
# def mean_confidence_interval(data):
#     m = np.mean(data)
#     err = scipy.stats.sem(data)  # calculates standard error of mean
#     return m, err, err
#
#
# def list_files_csv(path):
#     """
#     Get a list of all files and associated metadata from the csv data
#     """
#     print("list_files called")
#     r = []
#     for root, dirs, files in os.walk('.'):
#         for name in files:
#             filepath = os.path.join(root, name)
#             if ".csv" in name:
#                 # filepath = ./csv/rat4_shaping2/rat4_2019-02-20_11_23_33.csv
#                 # name = rat4_2019-02-20_11_23_33.csv
#                 meta_dict = {}
#                 meta_dict['rat_num'] = int(filepath[9])
#                 meta_dict['shaping_num'] = int(filepath[18])
#                 # TODO: change this to just year month day?
#                 meta_dict['datetime'] = csv_datestr_to_date_filename(filepath[-23:-4])
#                 meta_dict['filepath'] = filepath
#
#                 r.append(meta_dict)
#     return r
#
#
# def csv_to_dict_condensed(filepath, shape):
#     """
#     Given a filepath, convert the csv to a data dictionary, with keys
#     corresponding to column labes and values the list of all the columns.
#     """
#     with open(filepath, newline='') as csvfile:
#         spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#         header_row = next(spamreader)
#
#         # Deal with multiple delimiter types
#         if len(header_row) == 1:
#             header_row = header_row[0].split(',')
#
#         data = dict()
#         for col_name in header_row:
#             data[col_name] = []
#
#         cur = [0, 0, 0, 0, 0, 0]
#         sequence = dict()
#         for row in spamreader:
#             if len(row) == 0:
#                 # skip empty rows
#                 continue
#             if row[0] == 'time_stamp':
#                 # skip first row
#                 continue
#
#             vals = re.split(',| ', row[0])
#
#             # if (vals[1] == 'error'):
#
#             if (vals[1] == 'in_base' or vals[1] == 'already_in_base' or vals[1] == 'distractor_avoided'):
#                 new = [int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]
#                 if new[2] < 0:
#                     new[2] = 0
#                 mark = -1
#                 for i in range(len(cur)):
#                     if (new[i] - cur[i] > 0):
#                         sequence[vals[0]] = i
#                         mark = i
#                         break
#                 for i in range(len(cur)):
#                     cur[i] = new[i]
#
#             if (vals[1] == 'missed_trial'):
#                 sequence[vals[0]] = 0
#                 cur[0] += 1
#
#             for i, col_name in enumerate(header_row):
#                 data[col_name].append(vals[i])
#         # print(sequence)
#         return sequence  # data still holds the entire thing, but sequence is just a list of events with times
#
#
# def extract_consecutive_data(path, shape):
#     # print(path, shape)
#     size = 6
#     fullRecord = np.zeros(size)
#     data = csv_to_dict_condensed(path, shape)
#     # print(data)
#     counts = np.zeros((size, size))
#     order = []
#     time = 0
#     for i in data.keys():
#         order.append(data[i])
#         fullRecord[data[i]] += 1
#         time = csv_datestr_to_date_csvdata_2(i)
#     for i in range(1, len(order)):
#         counts[order[i]][order[i - 1]] += 1
#     return time, fullRecord, counts  # do i need order?
#
#
# def normalize(counts, size):
#     rowsum = np.zeros(size)
#     total = 0
#     for i in range(size):
#         for j in range(size):
#             rowsum[i] += counts[i][j]
#         total += rowsum[i]
#
#     twodcounts = []  # still contains unnormalized 2d counts, just isn't graphed right now
#     conditional2d = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             twodcounts.append(counts[i][j])
#             if counts[i][j] == 0:
#                 continue
#             conditional2d[i][j] = counts[i][j] * 1.0 / total  # normalizes into probabilities
#             conditional2d[i][j] = conditional2d[i][j] * 1.0 / (rowsum[i] / total)  # conditional probabilities
#     return conditional2d
#
#
# def compile_and_normalize(path, shape, rat):
#     size = 6
#     counts = np.zeros((size, size))  # counts consecutives
#     fullRecord = np.zeros(size)  # total number of m, e, l, c
#     segdict = {}  # stores by day, the number of m, e, l, c
#     for root, dirs, files in os.walk(path):
#         numfiles = len(files) - 1
#         files.sort()
#         count = 0
#         for name in files:
#             count += 1
#             filepath = os.path.join(root, name)
#             ratid = "rat" + str(rat)
#             if ratid not in filepath:
#                 continue
#             if ".csv" in name:
#                 time, fullRecord1, counts1 = extract_consecutive_data(filepath, shape)
#                 segdict[time] = fullRecord1
#                 # print(filepath, time, fullRecord1)
#                 for i in range(size):
#                     fullRecord[i] += fullRecord1[i]
#                     for j in range(size):
#                         counts[i][j] += counts1[i][j]
#
#     # print("segdict", segdict)
#     # print("fullRecord", fullRecord)
#     conditional2d = normalize(counts, size)
#
#     return segdict, fullRecord, conditional2d
#
#
# def graph_normalized_overtime(shape, train, pretrain):
#     size = 6
#
#     norm = {}
#     uncond = {}
#     timecounts = {}
#     totaldict = {}
#     avg = {}
#     std = {}
#     # this one's not fixed for AI format yet
#     # instead maybe i don't need to iterate through the rats, just count them
#     path = main_path + '/train_' + str(train) + '/reports/'
#     for root, dirs, files in os.walk(path):
#         for name in files:
#             filepath = os.path.join(root, name)
#             newname = name.split("_")
#             i = int(newname[0][3:])
#             if ".csv" in name:
#                 time, fullRecord, counts = extract_consecutive_data(filepath, shape)
#                 # print(i, filepath, time, fullRecord)
#                 fsum = sum(fullRecord)
#                 for j in range(size):
#                     fullRecord[j] /= fsum
#                 conditional = normalize(counts, size)
#                 if time not in norm:
#                     norm[time] = conditional
#                     timecounts[time] = 1
#                     uncond[time] = np.zeros((size, num_rats))
#                     totaldict[time] = np.zeros((size, size, num_rats))
#                     for j in range(size):
#                         for k in range(size):
#                             for l in range(num_rats):
#                                 totaldict[time][j][k][l] = -1
#                                 uncond[time][j][l] = -1
#                 else:
#                     timecounts[time] += 1
#                     for j in range(size):
#                         for k in range(size):
#                             norm[time][j][k] += conditional[j][k]
#                 for j in range(size):
#                     for k in range(size):
#                         totaldict[time][j][k][i] = conditional[j][k]
#                     uncond[time][j][i] = fullRecord[j]
#
#     # print(uncond)
#     for key in norm.keys():
#         for i in range(size):
#             for j in range(size):
#                 norm[key][i][j] /= timecounts[key]
#
#     # print("norm", norm)
#     # print("uncond", uncond)
#
#     print("to test independence:")
#     colors = ["red", "blue", "green", "black", "purple", "orange"]
#     labels = ['missed', 'early', 'late', 'distracted', 'distracted ok', 'correct']
#     wanted = [0, 1, 2, 3, 4, 5]
#
#     times = sorted(totaldict.keys())
#     uncondavg = {}
#     unconderr = {}
#     for key in sorted(totaldict.keys()):
#         uncondavg[key] = np.zeros(size)
#         unconderr[key] = np.zeros(size)
#         for i in range(size):
#             row = []
#             for j in range(num_rats):
#                 if (uncond[key][i][j] != -1):
#                     row.append(uncond[key][i][j])
#             uncondavg[key][i] = sum(row) / len(row)
#             unconderr[key][i] = np.std(row)
#     # print(uncondavg)
#     for i in range(size):
#         arravg = []
#         arrstd = []
#         for key in sorted(uncondavg.keys()):
#             arravg.append(uncondavg[key][i])
#             arrstd.append(unconderr[key][i])
#         plt.plot(times, arravg, marker='o', color=colors[i], label=labels[i])
#         plt.errorbar(times, arravg, arrstd, linestyle='None', color=colors[i])
#         plt.xticks(rotation=90)
#     plt.title('Probabilities of all rats from shaping ' + str(shape) + ' for training ' + str(
#         train) + ' with pretraining ' + str(pretrain))
#     plt.legend(loc='upper left')
#     plt.show()
#
#     for key in sorted(totaldict.keys()):
#         avg[key] = np.zeros((size, size))
#         std[key] = np.zeros((size, size))
#         for i in range(size):
#             for j in range(size):
#                 row = []
#                 for k in range(num_rats):
#                     if (totaldict[key][i][j][k] != -1):
#                         row.append(totaldict[key][i][j][k])
#                 avg[key][i][j] = sum(row) / len(row)
#                 std[key][i][j] = np.std(row)
#
#     for i in range(len(wanted)):
#         for j in range(size):
#             arravg = []
#             arrstd = []
#             for key in sorted(avg.keys()):
#                 arravg.append(avg[key][wanted[i]][j])
#                 arrstd.append(std[key][wanted[i]][j])
#             plt.plot(times, arravg, marker='o', color=colors[j], label=labels[j])
#             plt.errorbar(times, arravg, arrstd, linestyle='None', color=colors[j])
#             plt.xticks(rotation=90)
#         plt.title(
#             'Conditional probabilities given ' + labels[wanted[i]] + ' trial with stddev of shaping ' + str(shape))
#         plt.legend(loc='upper left')
#         plt.show()
#
#         split = [[], [], [], [], [], []]
#         for key in sorted(norm.keys()):
#             for j in range(size):
#                 split[j].append(norm[key][wanted[i]][j])
#
#         fig, ax = plt.subplots()
#         ax.stackplot(times, split[0], split[1], split[2], split[3], split[4], split[5], labels=labels)
#         ax.legend(loc='upper left')
#         plt.xticks(rotation=45)
#         plt.title('Conditional probabilities given ' + labels[wanted[i]] + ' trial of shaping ' + str(shape))
#         plt.show()
#
#
# def graph_normalized(path, shape):
#     segdict, fullRecord, conditional2d = compile_and_normalize(path, shape)
#     size = 6
#
#     conditional1d = []
#     for i in range(size):
#         for j in range(size):
#             conditional1d.append(conditional2d[i][j])
#
#     # plt.figure(figsize=(10,8))
#     possibles = []
#     for i in range(size):
#         for j in range(size):
#             possibles.append(str(i) + str(j))
#     index = np.arange(len(possibles))
#
#     # normalized
#     plt.bar(index, conditional1d)
#     plt.xlabel('Consecutive Steps', fontsize=5)
#     plt.ylabel('Counts', fontsize=5)
#     plt.xticks(index, possibles, fontsize=8, rotation=60)
#     plt.title('Normalized counts of matching steps for shaping ' + str(shape))
#     plt.show()
#
#     '''#un-normalized graph
#     plt.bar(index, twodcounts)
#     plt.xlabel('Consecutive Steps', fontsize=5)
#     plt.ylabel('Counts', fontsize=5)
#     plt.xticks(index, possibles, fontsize=8, rotation=30)
#     plt.title('Counts of matching steps for shaping ' + str(shape))
#     plt.show()'''
#
#     graph_total_outcomes(fullRecord, shape)
#
#
# def graph_total_outcomes(fullRecord, shape):
#     # total outcomes graph
#     outcomes = ('m', 'e', 'l', 'd', 'dok', 'c')
#
#     total = 0
#     for i in range(len(fullRecord)):
#         total += fullRecord[i]
#     index = np.arange(len(outcomes))
#     for i in range(len(fullRecord)):
#         fullRecord[i] = fullRecord[i] / total
#
#     plt.bar(index, fullRecord)
#     plt.xlabel('Outcomes', fontsize=5)
#     plt.ylabel('Counts', fontsize=5)
#     plt.xticks(index, outcomes, fontsize=8, rotation=30)
#     plt.title('Counts of outcomes')
#     plt.show()
#
#
# def graph_segmented_days(data,
#                          shape):  # data is dict with timestamps as keys and array of [m, e, l, c] as values for that day/time
#     '''
#     does sanity check so shows the progression of trials over all trials for 1 rat
#     '''
#     # print("data", data)
#     times = []
#     segmented = [[], [], [], [], [], []]
#     size = 6
#
#     for i in sorted(data.keys()):
#         times.append(i)
#         for j in range(size):
#             segmented[j].append(data[i][j])
#
#     # print("times", times)
#     # print("seg", segmented)
#     labels = ["correct", "distracted ok", "distracted", "early", "late", "missed"]
#
#     fig, ax = plt.subplots()
#     ax.stackplot(times, segmented[5], segmented[4], segmented[3], segmented[1], segmented[2], segmented[0],
#                  labels=labels)
#     ax.legend(loc='upper left')
#     plt.xticks(rotation=90)
#     plt.show()
#
#
# def combined_rats_total(shape, training, pretrain):
#     size = 6
#     sumdict = {}  # adds up for all days per trial type (this one isn't really needed)
#     totaldict = {}  # keeps values to calculate std dev
#     numcounts = {}  # keep track of how many rats had data for each day
#     for i in range(num_rats):
#         path1 = main_path + '/train_' + str(training) + '/reports/'
#         segdict, fullRecord, counts = compile_and_normalize(path1, shape, i)
#         for key in segdict.keys():
#             rsum = 0
#             for j in range(size):
#                 rsum += segdict[key][j]
#             for j in range(size):
#                 segdict[key][j] /= rsum
#         for key in sorted(segdict.keys()):
#             if key not in sumdict:
#                 numcounts[key] = 0
#                 sumdict[key] = np.zeros(size)
#                 totaldict[key] = np.zeros((size, num_rats))
#             numcounts[key] += 1
#             for j in range(size):
#                 sumdict[key][j] += segdict[key][j]
#                 totaldict[key][j][i - 1] = segdict[key][j]
#
#     for key in sorted(sumdict.keys()):
#         for j in range(size):
#             sumdict[key][j] /= numcounts[key]
#
#     colors = ["red", "blue", "green", "black", "purple", "orange"]
#     avgs = [[], [], [], [], [], []]
#     err = [[], [], [], [], [], []]
#     labels = ['missed', 'early', 'late', 'distracted', 'distracted ok', 'correct']
#
#     for i in range(size):
#         for key in sorted(totaldict.keys()):
#             avgs[i].append(sum(totaldict[key][i]) / numcounts[key])
#             err[i].append(np.std(totaldict[key][i]))
#
#     x = sorted(totaldict.keys())
#     for i in range(size):
#         plt.plot(x, avgs[i], marker='o', color=colors[i], label=labels[i])
#         plt.errorbar(x, avgs[i], err[i], linestyle='None', color=colors[i])
#         plt.xticks(rotation=90)
#     plt.legend(loc='upper left')
#     plt.title("Frequencies of trials for all rats in trial " + str(training) + ' with pretraining ' + str(pretrain))
#     plt.show()
#
#     # plt.plot(x, avgs[i], marker = 'o', color = colors[i], label = labels[i])
#     # plt.fill_between(x, avgs[i] - err[i], avgs[i] + err[i], linestyle='None', color = colors[i])
#     # plt.xticks(rotation=90)
#     # plt.show()
#
#     # fig,ax = plt.subplots()
#     # seaborn.lineplot('Date', 'Stock B', ci="sd", data=avgs[0],ax=ax)
#     # plt.show()
#
#     graph_segmented_days(sumdict, shape)
#
#
# def graph_segmented_perday(path, size):
#     # graphs it over the 30 minutes to test if the rats ever get hungry
#     # THIS ONE CAN'T BE DIRECTED USED FOR SIZE 6 SHAPING 2 SO AHVE TO CHANGE THAT
#
#     shape = 2
#     size = 6
#     totalpermin = np.zeros((size, 30))
#     mins = []
#     for i in range(30):
#         mins.append(i)
#     for root, dirs, files in os.walk(path):
#         for name in files:
#             filepath = os.path.join(root, name)
#             if ".csv" in name:
#                 # filepath = ./csv/rat4_shaping2/rat4_2019-02-20_11_23_33.csv
#                 # name = rat4_2019-02-20_11_23_33.csv
#                 times = []
#                 vals = []
#                 split = [[], [], [], [], [], []]
#                 permin = [[], [], [], [], [], []]
#                 data = csv_to_dict_condensed(filepath, shape)
#                 for key in data.keys():
#                     stime = key[11:16]
#                     times.append(stime)
#                     vals.append(data[key])
#                 start = times[0]
#                 cur = [int(start[:2]), int(start[3:])]
#                 end = times[len(times) - 1]
#                 done = [int(end[:2]), int(end[3:])]
#                 minutes = []
#                 while (cur != done):
#                     minutes.append(str(cur[0]) + '.' + str(cur[1]))
#                     cur[1] += 1
#                     if cur[1] == 60:
#                         cur[1] = 0
#                         cur[0] += 1
#                     for i in range(4):
#                         permin[i].append(0)
#                 minutes.append(str(cur[0]) + '.' + str(cur[1]))
#                 for i in range(size):
#                     permin[i].append(0)
#
#                 split = minutes[0].find('.')
#                 shour = int(minutes[0][:split])
#                 smin = int(minutes[0][split + 1:])
#                 for i in range(len(times)):
#                     csplit = times[i].find(':')
#                     chour = int(times[i][:csplit])
#                     cmin = int(times[i][csplit + 1:])
#                     posat = cmin - smin
#                     if posat < 0:
#                         posat = posat + 60
#                     permin[vals[i]][posat] += 1
#
#                 for i in range(len(permin)):
#                     minlen = min(len(totalpermin[0]), len(permin[0]))
#                     for j in range(minlen):
#                         totalpermin[i][j] += permin[i][j]
#
#     print("mins", mins)
#     print("totals", totalpermin)
#
#     fig, ax = plt.subplots()
#     labels = ["correct", "distracted ok", "distracted", "early", "late", "missed"]
#     ax.stackplot(mins, totalpermin[5], totalpermin[4], totalpermin[3], totalpermin[1], totalpermin[2], totalpermin[0],
#                  labels=labels)
#     ax.legend(loc='upper left')
#     plt.xticks(rotation=90)
#     plt.show()
#
#
# def record_trials_by_type(path, trial_objs, rat_nums=range(1, 7), shaping_nums=1):
#     """
#     Args:
#         path - main path to csv files
#         rat_nums - list of rat_nums the trial wants
#         shaping_nums - list of numbers for different shaping periods to apply this to
#         trial_funcs - list of functions, each of which is responsible for
#             keeping track of a type of trial as we iterate through a day of trials.
#     Returns:
#         dict - mapping type of trial to list of data describing each of the trials found.
#     """
#
#     data_dict = csv_to_dict(path)
#     event_dict = {}
#     for i, event in enumerate(data_dict['event']):
#         [obj.f(i, event, data_dict, event_dict) for obj in trial_objs]
#     return event_dict
#
#
# if __name__ == '__main__':
#     print("missed is 0, early is 1, late is 2, correct is 3")
#
#     shaping = 2
#     rat = 7
#
#     combined_rats_total(shaping, rat, True)
#     graph_normalized_overtime(shaping, rat, False)
#
#     path = main_path + '/train_' + str(rat) + '/reports/'
#     for i in [rat]:  # range(num_rats):
#         print(path, i)
#         segdict, fullRecord, counts = compile_and_normalize(path, 2, i)
#         # print(segdict)
#         print(fullRecord)
#         graph_segmented_days(segdict, 2)
#         break
#
#
#

