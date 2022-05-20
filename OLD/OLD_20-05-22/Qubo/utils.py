#!/usr/local/bin/python3

from .colors import colors
import csv
import time
import datetime


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)

def print_step(buffer, problem = None):
    if(problem==None):
        print(now()+" ["+colors.BOLD+colors.OKGREEN+"STEP"+colors.ENDC+"] " + buffer +"\n")
    else:
        print(now()+" ["+colors.BOLD+colors.OKBLUE+problem+colors.ENDC+"]"+" ["+colors.BOLD+colors.OKGREEN+"STEP"+colors.ENDC+"] " + buffer +"\n")
