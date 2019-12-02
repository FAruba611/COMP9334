# Name: COMP9334 Project2
# File: wrapper_file.py
# Version: V1.9
# Time: 2018.05.14

# Description:
# Basic M/M/s queueing module

#import simclass
import random
import queue
import math
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# -----------------------------------------------------------------------
# ---------- helper function ----------
# -------------------------------------
def average(lis):
    nsum = 0
    for i in range(len(lis)):
        nsum += lis[i]
    return nsum / len(lis)


# -----------------------------------------------------------------------
# ---------- experiment function ----------
# -----------------------------------------
def verify_exp_phase_distribute(scale, lmd=0.35, miu=1):
    # scale is 10000
    seed = eval(input('Enter a value for seed: '))
    random.seed(seed)
    gen_list_lmd = [random.random() for _ in range(scale)]
    gen_list_miu = [[random.random(),random.random(),random.random()] for _ in range(scale)]
    log_list_lmd = sorted([-math.log(1-gen_list_lmd[i]) / lmd for i in range(scale)])
    log_list_miu = sorted([sum([-math.log(1 - gen_list_miu[i][0]), -math.log(1 - gen_list_miu[i][1]), -math.log(1 - gen_list_miu[i][2])]) / miu for i in range(scale)])
    bins = 50
    max_v = max(log_list_lmd)
    min_v = min(log_list_lmd)

    interval = (max_v - min_v) / bins
    start = min_v
    split_list = list()
    y_list = list()

    #x_exp = np.linspace(min_v, max_v, 3000)

    while start <= max_v:
        #print("next interval = [{}, {}]".format(start, start + interval))
        #print(log_list)
        for i, item in enumerate(log_list_lmd):
            #print(item, start, start + interval)
            if item > start and item <= start + interval and len(log_list_lmd) == 1:
                split_list.append([item])
                y_list.append(i)
                break

            elif item > start + interval:
                split_list.append(log_list_lmd[0:i])
                #print(log_list[0:i])
                y_list.append(i)
                log_list_lmd = log_list_lmd[i:]
                break

        start += interval

    x_list = list()
    for sub_list in split_list:
        if len(sub_list) == 0:
            x_list.append(0)
        else:
            x_list.append(average(sub_list))

    #aim_function = [1800* lmd * math.exp(-  lmd * x_i) for x_i in x_list]
    plt.figure()
    plt.axvline(0)
    plt.plot(x_list, y_list)
    #plt.plot(x_list, aim_function)


def proper_tc_test(tc_hyp, w=480, nb_of_seed_test=30, lmd=0.35, miu=1, m=5, setup_time=5, tc_def=0.1, time_end=10000):
    t_diff_list = list()

    # \alpha = 1 - 0.95 = 0.05
    # t_n-1,1-\alpha/2 is at 1 - (0.05)/2 = 0.975
    tref_0_975_list30 = [12.706, 4.303, 3.182, 2.776, 2.571,
                        2.447, 2.365, 2.306, 2.262, 2.228,
                        2.201, 2.179, 2.160, 2.145, 2.131,
                        2.120, 2.110, 2.101, 2.093, 2.086,
                        2.080, 2.074, 2.069, 2.064, 2.060,
                        2.056, 2.052, 2.048, 2.045, 2.042
                        ]
    for s in range(nb_of_seed_test):
        base_arrival, base_service = process_random_mode(lmd, miu, time_end, seed_auto=s, switch="auto")
        # transient removal
        base_arrival = base_arrival[w:]
        base_service = base_service[w:]
        base_mrt = float(simulation('trace', base_arrival, base_service, m, setup_time, tc_def, time_end)[0])


        hypo_arrival, hypo_service = process_random_mode(lmd, miu, time_end, seed_auto=s, switch="auto")
        # transient removal
        hypo_arrival = hypo_arrival[w:]
        hypo_service = hypo_service[w:]
        hypo_mrt = float(simulation('trace', hypo_arrival, hypo_service, m, setup_time, tc_hyp, time_end)[0])
        t_diff_list.append(base_mrt - hypo_mrt)

    #t_diff_list = t_diff_list[w:]
    scale = len(t_diff_list)

    # calculate confidence interval
    avg_hat_diff = sum(t_diff_list) / scale

    tmp_sum = 0

    for diff_i in t_diff_list:
        square_error = pow(avg_hat_diff - diff_i, 2)
        tmp_sum += square_error
    stddev_hat_diff = math.sqrt(tmp_sum / (scale - 1))
    #print(tref_0_975_list30[nb_of_seed_test-1])
    # nb_of_seed_test should > 1
    floor = avg_hat_diff - ((stddev_hat_diff / math.sqrt(scale)) * tref_0_975_list30[nb_of_seed_test -2])
    ceil = avg_hat_diff + ((stddev_hat_diff / math.sqrt(scale)) * tref_0_975_list30[nb_of_seed_test -2])
    #print(t_diff_list)

    confidence_interval_95 = [floor, ceil]
    return confidence_interval_95


def transient_test(time_end, lmd=0.35, miu=1, m=5, setup_time=5, tc=0.1):
    # reproduce property
    arrival_list, service_list = process_random_mode(lmd, miu, time_end, seed_auto=1, switch="auto")
    series_of_arrival = list()
    series_of_service = list()
    for i in range(len(arrival_list)):
        series_of_arrival.append(arrival_list[0:i])
        series_of_service.append(service_list[0:i])
    if series_of_arrival[0] == []:
        if len(series_of_arrival) > 1:
            del series_of_arrival[0]
            del series_of_service[0]

            mrt_list = list()
            mrt_list.append(0)
            #print(len(series_of_service))
            for index, arrival_list in enumerate(series_of_arrival):
                #print("arrival = {}".format(arrival_list))
                #print("service = {}".format(series_of_service[index]))

                #print()
                current_mrt = simulation("trace", arrival_list, series_of_service[index], m, setup_time, tc, time_end)[0]
                mrt_list.append(float(current_mrt))

            # only when time_end is larger enough the following lines will be executed
            # w decided by observation
            w = 480
            if len(mrt_list) < w:
                w = len(mrt_list)
                theoretical_mrt = mrt_list[w - 1]
            else:
                theoretical_mrt = average(mrt_list[w:])
            plt.figure()

            plt.axhline(theoretical_mrt, linestyle="--")
            my_y_ticks = np.arange(0, 10, 1)
            #
            plt.plot(range(len(mrt_list)), mrt_list)
            plt.yticks(my_y_ticks)
            plt.show()


# -----------------------------------------------------------------------
# ---------- experiment function ----------
# -----------------------------------------
def process_random_mode(lmd, miu, time_end, seed_auto=1, switch="auto"):
    # generate the arrival and service list for trace simulation
    if switch == "auto":
        arrival_seed = seed_auto

    elif switch == "prompt":
        arrival_seed = eval(input('Enter a value for seed: '))

    time = 0
    arrival_list = list()
    service_list = list()
    random.seed(arrival_seed)

    while time < time_end:
        uk = random.random()
        sk1 = random.random()
        sk2 = random.random()
        sk3 = random.random()

        # print(arrival_random, tk1, tk2, tk3)
        current_arrival = time + -(math.log(1 - uk) / lmd)
        current_service = sum([-math.log(1 - sk1), -math.log(1 - sk2), -math.log(1 - sk3)]) / miu
        arrival_list.append(current_arrival)
        service_list.append(current_service)

        time = current_arrival
        # print(time)

    return arrival_list, service_list

# -----------------------------------------------------------------------

def wrapper():
    with open("num_tests.txt") as file:
        num_tests = int(file.readline())

    for test in range(num_tests):
        with open('mode_' + str(test + 1) + '.txt') as fm:
            mode = fm.readlines()[0]

            if mode == "random":
                with open('para_' + str(test + 1) + '.txt') as fpr:
                    para_list = [float(line.strip()) for line in fpr]
                    srv_num = int(para_list[0])
                    srv_setup_t = para_list[1]
                    srv_delay_t = para_list[2]
                    sim_end_time = para_list[3]
                with open('arrival_' + str(test + 1) + '.txt') as far:
                    lmd = float(far.readline())

                with open('service_' + str(test + 1) + '.txt') as fsr:
                    miu = float(fsr.readline())

                mrt, job_status = simulation(mode, lmd, miu, srv_num, srv_setup_t, srv_delay_t, sim_end_time)

            elif mode == "trace":
                sim_end_time = math.inf
                with open('para_' + str(test + 1) + '.txt') as fpt:
                    para_list = [float(line.strip()) for line in fpt]
                    srv_num = int(para_list[0])
                    srv_setup_t = para_list[1]
                    srv_delay_t = para_list[2]
                with open('arrival_' + str(test + 1) + '.txt') as fat:
                    arrival = [float(line.strip()) for line in fat]
                    #print(arrival)
                with open('service_' + str(test + 1) + '.txt') as fst:
                    service = [float(line.strip()) for line in fst]
                    #print(service)

                mrt, job_status = simulation(mode, arrival, service, srv_num, srv_setup_t, srv_delay_t, sim_end_time)

        depart_info_file = './departure_' + str(test + 1) + '.txt'
        mrt_file = './mrt_' + str(test + 1) + '.txt'
        with open(depart_info_file, 'w') as wd:
            for tuple in job_status:
                wd.write(tuple[0])
                wd.write("\t")
                wd.write(tuple[1])
                wd.write("\n")
        with open(mrt_file, 'w') as wm:
            wm.write(mrt)
            wm.write("\n")


def simulation(mode, arrival, service, m, setup_time, delayedoff_time, time_end):
    """
    :param mode: which in ['random','trace'], string type.
    :param arrival: It supplying arrival info to the program.
                    It has different meaning, which depends on mode.
    :param service: It supplying service info to the program.
                    It has different meaning, which depends on mode.
    :param m: number of servers
    :param setup_time: the setup time of a server or ... , positive float type.
    :param delayedoff_time: Initial value of countdown timer Tc.
    :param time_end: The end time of a simulation. This parameter is only relevant to random mode
    => example1: ('trace', [10, 20, 32, 33] , [1, 2, 3, 4] , 3, 50, 100, 170)
    => example niubility: simulation('trace', [10,18,20,23,28,32,33,34,35,57,86,92], [2,4,14,5,6,21,2,16,9,4,15,9],3,50,100,50)

    :return:
    """
    # -------- pre process ----------- #
    if mode not in ['random', 'trace']:
        print("wrong test case")
        return

    if mode == 'random':
        arrival,service = process_random_mode(arrival, service, time_end)

    # -------- start to simulate --------- #
    # Step1: Init the start state
    # Setting parameters

    # ----- clock
    master_clock = 0
    # ----- event
    next_job_arrive_time = arrival[0]
    next_job_arrive_service_time = service[0]

    next_job_depart_arrive_time = 0

    setup_time = setup_time
    delayedoff_time = delayedoff_time

    arrive_cnt = 1  # start from the second one to the last
    depart_cnt = 0
    nb_of_jobs = len(arrival)
    # ----- dispatcher
    buffer_queue = queue.Queue()  # this might be pop at setup mode, its service time is used in algorithm

    # ----- server
    svr_state_list = ['SVR_OFF'] * m

    svr_setup_done_t_list = [math.inf] * m
    svr_delay_done_t_list = [math.inf] * m
    job_depart_done_t_list = [math.inf] * m
    # svr_off_done_t_list = [math.inf] * m
    job_depart_done_arrive_t_list = [0] * m

    reject = 0
    job_status = list()

    response_time = 0

    # Step2: Iterator, continuing deal with the following steps
    while True:
        # ---------
        # Step1 => judge how to update the master clock by event
        # ---------
        # There are 4 states in the system, which is conducted by system jump point between different states
        #
        #  -----<-----         -------<------
        #  |         |         |            |
        # OFF ---> SETUP ---> BUSY ---> DELAYEDOFF , | | | | | |
        #  |                                |
        #  ----------------<-----------------
        # 1.job_arrive
        # 2.job_depart
        # 3.svr_setuped
        # 4.svr_delayoff

        time_list = [math.inf,
                     next_job_arrive_time,
                     min(job_depart_done_t_list),
                     min(svr_setup_done_t_list),
                     min(svr_delay_done_t_list)
                     ]
        # print("t_list = {}".format(time_list))
        next_event_time = min(time_list)
        if next_event_time == time_list[0]:
            next_event_type = "DEFAULT"  # end

        elif next_event_time == time_list[1]:
            next_event_type = "JOB_ARRIVE"

        elif next_event_time == time_list[2]:
            next_event_type = "JOB_DEPART"

        elif next_event_time == time_list[3]:  # waiting for a server to be setup
            next_event_type = "SVR_SETUP_DONE"

        elif next_event_time == time_list[4]:
            next_event_type = "SVR_DELAY_DONE"

        master_clock = next_event_time
        # print("Next event type update to ....{}".format(next_event_type))
        # print("master clock update to ....{}".format(master_clock))

        # Step2 => update essential variables and status
        nb_of_svr_off = svr_state_list.count("SVR_OFF")
        nb_of_svr_setup = svr_state_list.count("SVR_SETUP")
        nb_of_svr_busy = svr_state_list.count("SVR_BUSY")
        nb_of_svr_delay = svr_state_list.count("SVR_DELAYEDOFF")

        # Step3 => deal by different events
        # --- an arrival
        if next_event_type == "JOB_ARRIVE":
            # case1: nb_of_svr_delay > 0 (There exists delayed server in the system)
            # If there is at least a server in the DELAYEDOFF state, buffer send the top
            # arriving job to a particular server in the DELAYEDOFF state. The choice of the server
            # depends on the value of the countdown timer T_c of the servers at the time when the job arrives at
            # the dispatcher. Buffer will send the job to the server with the **highest** value in the
            # countdown timer. The selected server will cancel its countdown timer and change its state to BUSY

            # 1. we should find the index of maximum server Tc time
            # 2. update the next depart time for it by assign the job directly without setup step to the corresponding server

            #print(" ===== JOB ARRIVED ======")
            if nb_of_svr_delay > 0:
                #print("there exist delayedoff server")
                max_t = 0
                for time in svr_delay_done_t_list:
                    if time == math.inf:
                        continue
                    else:
                        if max_t < time:
                            max_t = time

                svr_index = svr_delay_done_t_list.index(max_t)
                job_depart_done_t_list[svr_index] = master_clock + next_job_arrive_service_time
                svr_state_list[svr_index] = "SVR_BUSY"
                job_depart_done_arrive_t_list[svr_index] = next_job_arrive_time
                svr_delay_done_t_list[svr_index] = math.inf

            # If there are no servers in the DELAYEDOFF state, then buffer will check whether there are servers in the OFF state.
            # If there is at least a server in the OFF state, then buffer will select one of them (maybe by order) and turn it on.
            # The state of the selected server will change to setup.
            # Buffer will put the job in at the end of the queue and mark the job.
            # A job in the queue is MARKED if it is waiting for a server to be setup. Hence, a simple consistency check
            # is that the number of MARKED jobs in the queue should equal to the number of servers in the SETUP state.
            else:
                if nb_of_svr_off > 0:
                    for svr_cnt, state in enumerate(svr_state_list):
                        if state == "SVR_OFF":
                            svr_setup_done_t_list[svr_cnt] = master_clock + setup_time
                            svr_state_list[svr_cnt] = "SVR_SETUP"

                            buffer_queue.put([next_job_arrive_time, next_job_arrive_service_time, "JOB_MARKED"])
                            break
                else:
                    buffer_queue.put([next_job_arrive_time, next_job_arrive_service_time, "JOB_UNMARKED"])

            # 3. update arrive cnt
            if arrive_cnt < nb_of_jobs:
                next_job_arrive_time = arrival[arrive_cnt]
                next_job_arrive_service_time = service[arrive_cnt]
                arrive_cnt += 1

            else:
                next_job_arrive_time = math.inf
                next_job_arrive_service_time = 0

        # ----------------------------------------------------
        elif next_event_type == "JOB_DEPART":
            #print(" ====== JOB_DEPART ====== ")
            svr_index = job_depart_done_t_list.index(master_clock)
            job_depart_done_t_list[svr_index] = math.inf
            next_job_depart_arrive_time = job_depart_done_arrive_t_list[svr_index]

            if mode == "random":
                if master_clock >= time_end:
                    break

            response_time += master_clock - next_job_depart_arrive_time
            
            a = Decimal(str(next_job_depart_arrive_time)).quantize(Decimal('0.000'))
            b = Decimal(str(master_clock)).quantize(Decimal('0.000'))
            job_status.append(( str(a), str(b) ))
            depart_cnt += 1
            buffer_size = buffer_queue.qsize()

            if buffer_size == 0:
                # If the dispatcher queue is empty, then the server will change its state from BUSY to DELAYEDOFF.
                # It will also start the countdown timer. The initial value of the countdown timer is Tc.
                # In this project, you can assume that Tc is deterministic.
                #svr_index = job_depart_done_t_list.index(master_clock)

                svr_state_list[svr_index] = "SVR_DELAYEDOFF"
                svr_delay_done_t_list[svr_index] = master_clock + delayedoff_time
                job_depart_done_t_list[svr_index] = math.inf
                next_job_depart_arrive_time = 0
                job_depart_done_arrive_t_list[svr_index] = next_job_depart_arrive_time

            else:
                # If there is at least one job at the queue, the server will take the job from the head of the queue.
                # The state of the server remains BUSY.
                # The other actions of buffer depends on whether the job that has been sent to the server
                # for processing is MARKED or UNMARKED.
                job_assign = buffer_queue.get()
                buffer_size = buffer_queue.qsize()
                svr_state_list[svr_index] = "SVR_BUSY"
                next_job_depart_service_time = job_assign[1]
                next_job_depart_arrive_time = job_assign[0]
                job_depart_done_arrive_t_list[svr_index] = next_job_depart_arrive_time

                #svr_index = job_depart_done_t_list.index(master_clock)
                job_depart_done_t_list[svr_index] = master_clock + next_job_depart_service_time

                if job_assign[2] == "JOB_MARKED":  # only handle when marked
                    # next_job_depart_arrive_time = job_assign[0]

                    tmplist = [buffer_queue.get() for _ in range(buffer_size)]
                    try:
                        tmplist[next(i for i, sublist in enumerate(tmplist) if sublist[2] == "JOB_UNMARKED")][
                            2] = "JOB_MARKED"

                    except StopIteration:
                        # No UNMARKED job
                        max_setup_done_t = 0
                        for time in svr_setup_done_t_list:
                            if time == math.inf:
                                continue
                            else:
                                if max_setup_done_t < time:
                                    max_setup_done_t = time
                        svr_index = svr_setup_done_t_list.index(max_setup_done_t)
                        svr_state_list[svr_index] = "SVR_OFF"
                        svr_setup_done_t_list[svr_index] = math.inf

                    for item in tmplist:
                        buffer_queue.put(item)
                    # get the first item in list which item[2] == "UM" and change it to "M"

        # ----------------------------------------------------
        elif next_event_type == "SVR_DELAY_DONE":
            #print(" ====== SVR_DELAY ====== ")
            svr_index = svr_delay_done_t_list.index(min(svr_delay_done_t_list))
            svr_state_list[svr_index] = "SVR_OFF"
            svr_delay_done_t_list[svr_index] = math.inf

        # ----------------------------------------------------
        elif next_event_type == "SVR_SETUP_DONE":
            #print(" ====== SVR_SETUP ====== ")
            job_assign = buffer_queue.get()
            svr_index = svr_setup_done_t_list.index(master_clock)
            svr_state_list[svr_index] = "SVR_BUSY"

            job_depart_done_t_list[svr_index] = master_clock + job_assign[1]
            job_depart_done_arrive_t_list[svr_index] = job_assign[0]
            svr_setup_done_t_list[svr_index] = math.inf

        # ----------------------------------------------------
        else:  # default
            # print(" ====== DEFAULT ====== ")
            # print(" END ")
            break

        show_info = [master_clock,
                       response_time,
                       [next_job_arrive_service_time, next_job_depart_arrive_time
                        ],
                       svr_state_list,
                       job_depart_done_arrive_t_list,
                       [
                           next_job_arrive_time,
                           svr_setup_done_t_list,
                           job_depart_done_t_list,
                           svr_delay_done_t_list,
                       ],
                       buffer_queue.queue
                       ]
        # print(show_info)
        # print()
        # Update status list
    
    mean_response_time = str(Decimal(str(response_time / depart_cnt)).quantize(Decimal('0.000')))
    print("+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
    print("=== new test ===")
    print("== Tuple of job arrive time and depart time :")
    print("== {}".format(job_status))
    print("== MRT :")
    print("== {}".format(mean_response_time))
    print("+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
    print()

    #write_wrapper
    return [mean_response_time, job_status]



#wrapper()
