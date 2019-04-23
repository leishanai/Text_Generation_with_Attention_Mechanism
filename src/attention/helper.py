import math
import time
import datetime



def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm-%ds' % (m, s)

def timer(since, percent):
    now = time.time()
    s = now - since # time elapsed
    es = s / (percent) # total time estimated
    rs = es - s # time left
    return "Time_elapsed:%s - Time_left: %s" % (as_minutes(s), as_minutes(rs))