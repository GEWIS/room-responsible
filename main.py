# Person class
import copy
import os.path
import sys
from datetime import datetime


class Person:
    def __init__(self, name):
        self.name = name
        self.weekcounter = 0
        self.morningweekcounter = 0
        self.afternoonweekcounter = 0
        self.totalcounter = 0
        self.morningcounter = 0
        self.afternooncounter = 0
        self.calendar = []
        self.non_busy = 0
        self.busy = 0
        self.ingeroosterdochtend = 0
        self.ingeroosterdmiddag = 0
        self.is_board = False
        self.max_shifts = -1

    def set_board(self, val):
        self.is_board = val

    def set_max_shifts(self, number):
        self.max_shifts = number

    def __str__(self):
        return f'Person({self.name}, {self.is_board}, {self.max_shifts})'

    def get_name(self):
        return self.name


class Date:
    def __init__(self, exams, is_monday, date):
        self.exams = exams
        self.is_monday = is_monday
        self.shifts = []
        self.date = date
        if len(DATES) > 0:
            self._last_date = DATES[-1]
        else:
            self._last_date = None
        if len(DATES) > 1:
            self._second_last_date = DATES[-2]
        else:
            self._second_last_date = None
        if len(DATES) > 2:
            self._third_last_date = DATES[-3]
        else:
            self._third_last_date = None

    def add_shift(self, shift):
        self.shifts.append(shift)

    def is_afternoon(self):
        return self.afternoon

    def is_exams(self):
        return self.exams

    def get_shifts(self):
        return self.shifts

    def __str__(self):
        string = f'Date({self.date}), consisting of shifts: \n'
        for i in self.shifts:
            string += f'- {str(i)}\n' # str(i) + '\n'
        return string


class Shift:
    def __init__(self, start, end, indicator):
        self.start = datetime.strptime(start, "%H:%M:%S")
        self.end = datetime.strptime(end, "%H:%M:%S")
        self.indicator = indicator
        self.available_people = []
        self.assigned_people = []

    def __str__(self):
        string = f'Shift ({self.indicator}, {datetime.strftime(self.start, "%H:%M:%S")} - {datetime.strftime(self.end, "%H:%M:%S")}), filled by: '
        for i in self.available_people:
            string += f'{i.get_name()}, '
        return string

    def add_available_person(self, person):
        self.available_people.append(person)

    def get_indicator(self):
        return self.indicator

    def assign_person(self, person):
        self.assigned_people.append(person)

DATES = []
PERSONS = []
SHIFTS = []
NO_ONE = Person("Get Room Responsible")


def read_availabilities(csv_name):
    global SHIFTS
    global PERSONS
    global DATES
    with open(csv_name, 'r') as file:
        index = 0

        # read all lines
        for line in file:
            # read first line, which are the shifts
            if index == 0:
                shifts = list(filter(None, line.rstrip().split(";")))
                for i in range(int(len(shifts) / 3)):
                    SHIFTS.append(Shift(shifts[i * 3], shifts[i * 3 + 1], shifts[i * 3 + 2]))
            elif index == 1:
                persons = list(filter(None, line.rstrip().split(";")))
                for i in range(2, len(persons)):
                    PERSONS.append(Person(persons[i]))
            elif index == 2:
                max_shifts = list(filter(None, line.rstrip().split(";")))
                for i in range(1, len(max_shifts)):
                    PERSONS[i - 1].set_max_shifts(max_shifts[i])
            elif index == 3:
                board = list(filter(None, line.rstrip().split(";")))
                for i in range(1, len(board)):
                    PERSONS[i - 1].set_board(board[i])
            else:
                data = line.rstrip().split(";")
                DATES.append(Date(int(data[2]), int(data[1]), datetime.strptime(data[0], "%m/%d/%Y")))
                availabilities = line.split(';')[(len(PERSONS) - 1):]
                for i in SHIFTS:
                    DATES[index - 4].add_shift(copy.deepcopy(i))
                for i, v in enumerate(availabilities):

                    for j in DATES[index - 4].get_shifts():
                        if j.get_indicator() in v:
                            j.add_available_person(PERSONS[i])
            index += 1
        for date in DATES:
            print(date)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1] if ".csv" in sys.argv[1] else sys.argv[1] + ".csv"
        if os.path.isfile(file_name):
            read_availabilities(file_name)
