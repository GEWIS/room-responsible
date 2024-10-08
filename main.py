# Person class
import copy
import os.path
from datetime import datetime
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from icalendar import Calendar, Event
import random
import numpy
import argparse

class Person:
    def __init__(self, name):
        self.name = name
        self.calendar = []
        self.non_busy = 0
        self.busy = 0
        self.is_board = False
        self.max_shifts = -1
        self.bin_preference = []
        self.bin_assign = []
        self.shift_assigned = {}
        for i in SHIFTS:
            self.shift_assigned[i.get_indicator()] = 0
        self.assigned = 0
        self.available = 0
        self.calendar = Calendar()

    def set_board(self, val):
        self.is_board = val

    def set_max_shifts(self, number):
        self.max_shifts = number

    def __str__(self):
        return f'Person({self.name}, {self.is_board}, {self.max_shifts})'

    def add_indicated_shift(self, indicator):
        self.shift_assigned[indicator] += 1

    def get_name(self):
        return self.name

    def get_available(self):
        return self.available

    def get_total(self):
        return self.assigned

    def set_bin_preference(self, preferred_list):
        self.bin_preference = preferred_list
        self.available = sum(preferred_list)

    def set_bin_assignment(self, assignment_list):
        self.bin_assign = assignment_list

    def get_bin_preference(self):
        return self.bin_preference

    def get_bin_assignment(self):
        return self.bin_assign

    def get_is_board(self):
        return self.is_board

    def get_max_shifts(self):
        return self.max_shifts

    def get_indicated_shift(self, indicator):
        return self.shift_assigned[indicator]

    def assign_from_bin(self):
        for i in range(len(DATES)):
            for j in range(len(SHIFTS)):
                if self.bin_assign[i * len(SHIFTS) + j] == 1:
                    shift = DATES[i].get_shifts()[j]
                    shift.assign_person(self)
                    self.add_indicated_shift(SHIFTS[j].get_indicator())
                    self.assigned += 1
        self.available = sum(self.bin_assign)

    def get_calendar(self):
        return self.calendar

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

    def is_exams(self):
        return self.exams

    def get_shifts(self):
        return self.shifts

    def get_date(self):
        return self.date

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
        for i in self.assigned_people:
            string += f'{i.get_name()}, '
        return string

    def add_available_person(self, person):
        self.available_people.append(person)

    def assign_person(self, person):
        self.assigned_people.append(person)

    def get_indicator(self):
        return self.indicator

    def get_assigned_persons(self):
        return self.assigned_people

    def get_start_time(self):
        return self.start

    def get_end_time(self):
        return self.end

class RoomResponsibleSchedulingProblem:
    """This class encapsulates the Nurse Scheduling problem
    """

    def __init__(self):

        self.people = PERSONS

        self.people_per_shift = 2

    def __len__(self):
        return len(DATES) * len(SHIFTS) * len(PERSONS)

    def get_room_responsible_shifts(self, schedule):
        shifts_per_person = self.__len__() // len(self.people)
        person_shift_dict = {}
        shift_index = 0

        for person in self.people:
            person_shift_dict[person.get_name()] = schedule[shift_index:shift_index + shifts_per_person]
            shift_index += shifts_per_person

        return person_shift_dict

    def get_cost(self, schedule):
        if len(schedule) != self.__len__():
            raise ValueError(f'Size of schedule list should be equal to: {self.__len__()}, instead got: {len(schedule)} for schedule {schedule} and {len(PERSONS), len(SHIFTS), len(DATES)}')

        shifts_dict = self.get_room_responsible_shifts(schedule)

        board_violations = self.count_board_violations(schedule)
        max_shift_violations = self.count_max_shift_violations(shifts_dict)
        people_per_shift_violations = self.count_people_per_shift_violations(shifts_dict)
        non_board_violations = self.count_non_board_violations(schedule)
        consecutive_shift_violations = self.count_consecutive_shift_violations(shifts_dict)
        preference_violations = self.count_preference_violations(shifts_dict)

        violations = [board_violations, max_shift_violations, people_per_shift_violations, non_board_violations,
                      consecutive_shift_violations, preference_violations]
        weights = [30, 25, 100, 10, 1, 200]
        return sum(v * w for v, w in zip(violations, weights))

    def count_board_violations(self, schedule):
        violations = 0
        shifts_per_person = self.__len__() // len(self.people)
        for i in range(shifts_per_person):
            board_assigned = False
            for j in range(len(self.people)):
                if schedule[j * shifts_per_person + i] == 1 and PERSONS[j].get_is_board():
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
        return violations

    def count_people_per_shift_violations(self, personalized_schedule):
        return sum(1 for shift in zip(*personalized_schedule.values()) if sum(shift) != self.people_per_shift)

    def count_non_board_violations(self, schedule):
        violations = 0
        shifts_per_person = self.__len__() // len(self.people)
        for i in range(shifts_per_person):
            non_board_assigned = False
            for j in range(len(self.people)):
                if schedule[j * shifts_per_person + i] == 1 and not bool(PERSONS[j].get_is_board()):
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
            preferences = i.get_bin_preference()
            for j in range(len(personalized_schedule[i.get_name()])):
                if preferences[j] == 0 and personalized_schedule[i.get_name()][j] == 1:
                    violations += 1

        return violations

    def print_schedule_info(self, schedule):

        shifts_dict = self.get_room_responsible_shifts(schedule)

        print("Schedule for each room responsible")
        for person in shifts_dict:
            print(f'{person} : {shifts_dict[person]}')
            person_object = get_person_by_name(person)
            person_object.set_bin_assignment(shifts_dict[person])

        print(f'Board violations: {self.count_board_violations(schedule)} \n')
        print(f'Weekly Shift Violations: {self.count_max_shift_violations(shifts_dict)} \n')
        print(f'People per shift violations: {self.count_people_per_shift_violations(shifts_dict)} \n')
        print(f'Non board violations: {self.count_non_board_violations(schedule)} \n')
        print(f'Consecutive shift violations: {self.count_consecutive_shift_violations(shifts_dict)} \n')
        print(f'Preference violations {self.count_preference_violations(shifts_dict)} \n')
        print("Shifts per person")
        for person in PERSONS:
            print(f'{person.get_name()}: {sum(shifts_dict[person.get_name()])}')

def ea_simple_with_elitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
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

def get_person_by_name(name):
    global PERSONS

    for i in PERSONS:
        if i.get_name() == name:
            return i

def print_results():
    global NO_ONE

    # Write resulting shifts to file with UTF-8 encoding
    with open('OpenhoudenResults.csv', 'w', encoding='utf-8-sig') as file:
        file.write(f'Subject, Start Date, Start Time, End Date, End Time \n')
        for date in DATES:
            for shift in date.get_shifts():
                room_responsible_shift = ""
                while len(shift.get_assigned_persons()) < 2:
                    shift.assign_person(copy.deepcopy(NO_ONE))
                room_responsible_shift += f'{shift.get_assigned_persons()[0].get_name()} & {shift.get_assigned_persons()[1].get_name()},'
                room_responsible_shift += f'{datetime.strftime(date.get_date(), "%d/%m/%Y")}, {datetime.strftime(shift.get_start_time(), "%H:%M:%S")}, {datetime.strftime(date.get_date(), "%d/%m/%Y")}, {datetime.strftime(shift.get_end_time(), "%H:%M:%S")} \n'
                file.write(room_responsible_shift)

    with open("OpenhouderStats.csv", "w", encoding='utf-8-sig') as file:  # Use UTF-8 encoding
        file.write("STATS\n")

        # Create headers for each person
        headers = f'Shift\\Person,' + ','.join([person.get_name() for person in PERSONS]) + '\n'
        file.write(headers)

        # Write availability and total assignments
        file.write('Available,' + ','.join(str(person.get_available()) for person in PERSONS) + '\n')
        file.write('Total,' + ','.join(str(person.get_total()) for person in PERSONS) + '\n')

        # Write shift assignment information
        for shift in SHIFTS:
            shift_row = [shift.get_indicator()]
            for person in PERSONS:
                shift_row.append(str(person.get_indicated_shift(shift.get_indicator())))
            file.write(','.join(shift_row) + '\n')

    cal = Calendar()

    for date in DATES:
        for shift in date.get_shifts():
            assigned_persons = shift.get_assigned_persons()
            event = Event()
            event.add('summary', ' & '.join([person.get_name() for person in assigned_persons]))
            event.add('dtstart', datetime.combine(date.get_date(), shift.get_start_time().time()))
            event.add('dtend', datetime.combine(date.get_date(), shift.get_end_time().time()))
            event.add('dtstamp', datetime.now())
            event.add('location', 'MF 3.155')
            event.add('description', 'Room Responsible Shift')

            cal.add_component(event)
            for person in assigned_persons:
                person.get_calendar().add_component(event)


    with open('OpenhoudenSchedule.ics', 'wb') as file:
        file.write(cal.to_ical())
    if not os.path.exists('schedules'):
        os.makedirs('schedules')
    for person in PERSONS:
        with open(f'schedules/Openhouden{person.get_name()}.ics', 'wb') as file:
            file.write(person.get_calendar().to_ical())

    print("iCalendar files created succesfully")

DATES = []
PERSONS = []
SHIFTS = []
NO_ONE = Person("Get Room Responsible")
file_name = "availability.csv"

def read_availabilities(csv_name):
    global SHIFTS
    global PERSONS
    global DATES
    PERSONS = []
    DATES = []
    SHIFTS = []
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
                    PERSONS[i - 1].set_max_shifts(int(max_shifts[i]))
            elif index == 3:
                board = list(filter(None, line.rstrip().split(";")))
                for i in range(1, len(board)):
                    PERSONS[i - 1].set_board(int(board[i]))
            else:
                data = line.rstrip().split(";")
                DATES.append(Date(int(data[2]), int(data[1]), datetime.strptime(data[0], "%m/%d/%Y")))
                availabilities = line.split(';')[3:]
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



# Genetic Algorithm constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.4  # probability for mutating an individual
max_generations = 2000
HALL_OF_FAME_SIZE = 30
parser = argparse.ArgumentParser(description="List of arguments")
parser.add_argument("-g", "--generations", help = "How many generations should be run")
parser.add_argument("-i", "--input", help="Input file path")
# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

if __name__ == "__main__":

    args = parser.parse_args()
    if args.generations:
        max_generations = int(args.generations)
    if args.input:
        file_name = args.input

    if os.path.isfile(file_name):
        read_availabilities(file_name)
        rrsp = RoomResponsibleSchedulingProblem()

        # define a single objective, maximizing fitness strategy:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # create the Individual class based on list:
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # create an operator that randomly returns 0 or 1:
        toolbox.register("zeroOrOne", random.randint, 0, 1)

        # create the individual operator to fill up an Individual instance:
        toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(rrsp))

        # create the population operator to generate a list of individuals:
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


        # fitness calculation
        def get_cost(individual):
            return rrsp.get_cost(individual),  # return a tuple


        toolbox.register("evaluate", get_cost)

        # genetic operators:
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(rrsp))

        # create initial population (generation 0):
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min)
        stats.register("avg", numpy.mean)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        # perform the Genetic Algorithm flow with hof feature added:
        population, logbook = ea_simple_with_elitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                     ngen=max_generations, stats=stats, halloffame=hof, verbose=True)

        # print best solution found:
        best = hof.items[0]
        print("-- Best Individual = ", best)
        print("-- Best Fitness = ", best.fitness.values[0])
        print()
        print("-- Schedule = ")
        rrsp.print_schedule_info(best)

        for i in PERSONS:
            i.assign_from_bin()

        print_results()