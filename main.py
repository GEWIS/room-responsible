# Person class
import copy
import os.path
import sys
from datetime import datetime
from deap import tools
from deap import algorithms
import numpy as np


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


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
        self.bin_preference = []
        self.bin_assign = []

    def set_board(self, val):
        self.is_board = val

    def set_max_shifts(self, number):
        self.max_shifts = number

    def __str__(self):
        return f'Person({self.name}, {self.is_board}, {self.max_shifts})'

    def get_name(self):
        return self.name

    def set_bin_preference(self, preferred_list):
        self.bin_preference = preferred_list

    def set_bin_assignment(self, assignment_list):
        self.bin_assign = assignment_list

    def get_bin_preference(self):
        return self.bin_preference

    def get_bin_assignment(self):
        return self.bin_assign

    def is_board(self):
        return self.is_board

    def get_max_shifts(self):
        return self.max_shifts


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
            string += f'- {str(i)}\n'
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


class RoomResponsibleSchedulingProblem:
    """This class encapsulates the Nurse Scheduling problem
    """

    def __init__(self, hard_constraint_penalty):
        self.hard_constraint_penalty = hard_constraint_penalty

        self.people = PERSONS

        self.people_per_shift = 2

    def __len__(self):
        return len(DATES) * len(SHIFTS)

    def get_room_responsible_shifts(self, schedule):
        shifts_per_person = self.__len__() // len(self.people)
        person_shift_dict = {}
        shift_index = 0

        for person in self.people:
            person_shift_dict[person.get_name()] = schedule[shift_index:shift_index + shifts_per_person]
            shift_index += shifts_per_person

        return person_shift_dict

    def count_board_violations(self, schedule):
        violations = 0
        shifts_per_person = self.__len__() // len(self.people)
        for i in range(shifts_per_person):
            board_assigned = False
            for j in range(len(self.people)):
                if schedule[j*shifts_per_person+i] == 1 and PERSONS[j].is_board():
                    board_assigned = True
            if not board_assigned:
                violations += 1
        return violations

    def count_max_shift_violations(self, personalized_schedule):
        violations = 0
        for i in self.people:
            shift_count = sum(personalized_schedule[i.get_name()])
            max_shifts = i.get_max_shifts()
            if max_shifts == -1:
                continue
            violations += max([0, (shift_count - max_shifts // 4)])

    def count_people_per_shift_violations(self, personalized_schedule):
        return sum(1 for shift in zip(*personalized_schedule.values()) if sum(shift) != self.people_per_shift)

    def count_non_board_violations(self, schedule):
        violations = 0
        shifts_per_person = self.__len__() // len(self.people)
        for i in range(shifts_per_person):
            non_board_assigned = False
            for j in range(len(self.people)):
                if schedule[j*shifts_per_person+i] == 1 and not PERSONS[j].is_board():
                    non_board_assigned = True
            if not non_board_assigned:
                violations += 1
        return violations

    def count_consecutive_shift_violations(self, personalized_schedule):
        violations = 0
        for shifts in personalized_schedule.values():
            for shift1, shift2 in zip(shifts, shifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def count_preference_violations(self, personalized_schedule):
        violations = 0
        for i in PERSONS:
            preferences = i.get_bin_preference
            for j in range(len(personalized_schedule[i.get_name()])):
                if preferences[j] == 0 and personalized_schedule[i.get_name][j] == 1:
                    violations += 1

        return violations

    def print_schedule_info(self, schedule):

        shifts_dict = self.get_room_responsible_shifts(schedule)

        print("Schedule for each room responsible")
        for person in shifts_dict:
            print(f'{person} : {shifts_dict[person]}')

        print(f'Board violations: {self.count_board_violations(schedule)} \n')
        print(f'Weekly Shift Violations: {self.count_max_shift_violations(shifts_dict)} \n')
        print(f'People per shift violations: {self.count_people_per_shift_violations(shifts_dict)} \n')
        print(f'Non board violations: {self.count_non_board_violations(shifts_dict)} \n')
        print(f'Consecutive shift violations: {self.count_consecutive_shift_violations(shifts_dict)} \n')
        print(f'Preference violations {self.count_preference_violations(shifts_dict)} \n')



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
                            PERSONS[i].bin_preference.append(1)
                        else:
                            PERSONS[i].bin_preference.append(0)
            index += 1
        # for date in DATES:
        #     print(date)
        for person in PERSONS:
            print(len(person.bin_preference))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1] if ".csv" in sys.argv[1] else sys.argv[1] + ".csv"
        if os.path.isfile(file_name):
            read_availabilities(file_name)
