# Person claso
import re

import copy
import os.path
from datetime import datetime
# from deap import algorithms
# from deap import base
# from deap import creator
# from deap import tools
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

    def increment_assigned(self):
        self.assigned += 1

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
    def __init__(self, start, end, indicator, weight):
        self.start = datetime.strptime(start, "%H:%M:%S")
        self.end = datetime.strptime(end, "%H:%M:%S")
        self.indicator = indicator
        self.available_people = []
        self.assigned_people = []
        self.weight = weight

    def __str__(self):
        string = f'Shift ({self.indicator}, {datetime.strftime(self.start, "%H:%M:%S")} - {datetime.strftime(self.end, "%H:%M:%S")}), filled by: '
        for i in self.assigned_people:
            string += f'{i.get_name()}, '
        return string

    def set_weight(self, val):
        self.weight = val

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

    def get_weight(self):
        return self.weight


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
        weights = [3, 10, 10, 1, 0, 20]
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


import gurobipy as gp
from gurobipy import GRB
def solve():
    global BIN_WEIGHTS
    SHIFTSTOT = len(SHIFTS) * len(DATES)

    N = [] # People with max shifts assigned
    B = [] # Board members
    BI = [] # Board members with infinite shifts

    m = gp.Model("mip1")

    for j, person in enumerate(PERSONS):
        if person.get_is_board():
            B.append(j)
            if person.get_max_shifts() == -1:
                BI.append(j)
        if person.get_max_shifts() != -1:
            N.append(j)


    # Variables
    r = m.addMVar(shape=SHIFTSTOT, vtype=GRB.INTEGER, name="r")
    b = m.addMVar(shape=SHIFTSTOT, vtype=GRB.INTEGER, name="b")
    x = m.addMVar(shape=(SHIFTSTOT, len(PERSONS)), vtype=GRB.BINARY, name="x")
    l = m.addMVar(shape=SHIFTSTOT//3, vtype=GRB.INTEGER, name="l")
    n = m.addMVar(shape=len(N), vtype=GRB.INTEGER, name="n")
    bv = m.addMVar(shape=len(B), vtype=GRB.INTEGER, name="bv")

    # Availability constraint
    for i in range(SHIFTSTOT):
        for j, person in enumerate(PERSONS):
            m.addConstr(x[i][j] <= person.get_bin_preference()[i], f"available_{i}_{j}")

    for i in range(SHIFTSTOT):
        # Extra variable for people assigned to shift
        m.addConstr(grsum(x[i]) == r[i], f"rge_{i}")
        m.addConstr(r[i] <= 2, f"ass_1_{i}")

        # All shifts have at least one board member
        sum = gp.LinExpr()
        for j in B:
            sum += x[i][j]
        m.addConstr(sum >= b[i], f"boardav_{i}")
        m.addConstr(sum >= b[i], f"boardav_{i}")

    for i in range(0, SHIFTSTOT, 3):
        l1 = rowmult(x[i], x[i+1])
        l2 = rowmult(x[i+2], x[i+1])
        l3 = rowmult(l1, l2)
        l4 = grsum(l1) + grsum(l2) - grsum(l3)
        m.addConstr(l[i // 3] == l4, f"l_{i//3}")
        # m.addConstr(l[i // 3] <= 1, f"ltop_{i//3}")
         
    # People with a max shifts get maximum their max shifts. 
    for i, j in enumerate(N):
        person = PERSONS[j]
        m.addConstr(wegrsum(get_column(x, j), BIN_WEIGHTS) == n[i], f"n_{j}")
        m.addConstr(n[i] <= person.get_max_shifts(), f"maxshift_{j}")

    for i, j in enumerate(B):
        m.addConstr(wegrsum(get_column(x, j), BIN_WEIGHTS) == bv[i], f"bv_{j}")

    # Mean (aux variable)
    mean = m.addVar(lb=-GRB.INFINITY, name="mean")

    # Constraint for mean
    m.addConstr(mean == (1/len(B)) * gp.quicksum(bv[i] for i in B))

    # Variance expression
    variance = (1/len(B)) * gp.quicksum((bv[i] - mean)*(bv[i] - mean) for i in B)
    # Objective: minimize variance
    m.ModelSense = GRB.MAXIMIZE

    m.setObjective(3 * grsum(r) + grsum(b) + grsum(n) + grsum(l) - variance, GRB.MAXIMIZE)

    # Set maximization objectives
    # m.setObjectiveN(grsum(r), 0, 0)
    # m.setObjectiveN(grsum(b), 1, 1)
    # m.setObjectiveN(grsum(n), 2, 2)
    # m.setObjectiveN(-variance, 3, 3)
    # m.setObjectiveN(-grsum(var), 2, 2)

    m.optimize()

    # print(m.display())
    
    for v in m.getVars():

        try: 
            print('%s %g' % (v.VarName, v.X))
        except:
            print('%s' % (v.VarName))

    bin_prefs = [[0 for _ in range(SHIFTSTOT)] for _ in range(len(PERSONS))]
    for v in m.getVars():
        index = re.split(r'[\[\],]+', v.VarName)
        if (index[0] == "x"):
            shift, person = int(index[1]), int(index[2])
            bin_prefs[person][shift] = int(v.X)

    for i in range(len(PERSONS)):
        PERSONS[i].set_bin_assignment(bin_prefs[i])
        print(PERSONS[i].get_bin_assignment())
        print(len([x for i, x in enumerate(PERSONS[i].get_bin_assignment()) if i % 3 == 1]))

def grsum(x):
    obj = gp.LinExpr()
    for expr in x:
        obj += expr
    return obj

def rowmult(x1, x2):
    # obj = gp.LinExpr()
    obj = []
    for i, j in zip(x1, x2):
        obj.append(i * j)
    return obj

def wegrsum(x, weights):
    obj = gp.LinExpr()
    for i, expr in enumerate(x):
        obj += weights[i] * expr
    return obj

def get_column(x, i) -> list:
    return [row[i] for row in x]


    # TO_SOLVE = []
    # # First check if some dates have to be filled in some way.
    # for shift in SHIFTS:
    #     av = len(shift.available_people)
    #     match av:
    #         case 0:
    #             shift.assign_person(copy.deepcopy(NO_ONE))
    #         case 1:
    #             shift.assign_person(shift.available_people[0])
    #             shift.assign_person(copy.deepcopy(NO_ONE))
    #
    #             shift.available_people[0].increment_assigned()
    #         case 2:
    #             shift.assign_person(shift.available_people[0])
    #             shift.assign_person(shift.available_people[1])
    #
    #             shift.available_people[0].increment_assigned()
    #             shift.available_people[1].increment_assigned()
    #         case _:
    #             TO_SOLVE.append(shift)

    

    # Now that that has been done, we should fill up per week (maximum shifts)
    # Then we fill any slots with RRC members that want to fill in. 
        # Sidenote, you have to check that it works out in the end. 


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
BIN_WEIGHTS = []
NO_ONE = Person("Get Room Responsible")
file_name = "availability.csv"

def line_to_list(line: str): 
    return list(filter(None, line.rstrip().split(";")))

def read_availabilities(csv_name):
    global SHIFTS
    global PERSONS
    global DATES
    PERSONS = []
    DATES = []
    SHIFTS = []

    # The amount of cells one Shift takes in the csv file
    SHIFTCSV = 4
    # How many datacolumns each date has (date + is_exam currently) 
    DATEDATA = 2
    # How man rows of information before the dates start 
    DATEDATASTART = 4

    with open(csv_name, 'r') as file:
        # Read all lines
        for index, line in enumerate(file):
            # Read first line, which are the shifts
            if index == 0:
                shifts = line_to_list(line)
                for i in range(int(len(shifts) / SHIFTCSV)):
                    SHIFTS.append(Shift(*[shifts[i * SHIFTCSV + j] for j in range(SHIFTCSV)]))
            elif index == 1:
                persons = line_to_list(line)
                for i in range(1, len(persons)):
                    PERSONS.append(Person(persons[i]))
            elif index == 2:
                max_shifts = line_to_list(line)
                for i in range(1, len(max_shifts)):
                    PERSONS[i - 1].set_max_shifts(int(max_shifts[i]) if int(max_shifts[i]) == -1 else int(max_shifts[i]) * 4)
            elif index == 3:
                board = line_to_list(line)
                for i in range(1, len(board)):
                    PERSONS[i - 1].set_board(int(board[i]))
            else:
                data = line.rstrip().split(";")
                dt = datetime.strptime(data[0], "%d/%m/%Y")
                DATES.append(Date(exams = int(data[1]), is_monday = dt.weekday() == 0, date = dt))

                availabilities = line.split(';')[DATEDATA:]
                for i in SHIFTS:
                    DATES[index - DATEDATASTART].add_shift(copy.deepcopy(i))
                    BIN_WEIGHTS.append(i.get_weight())

                for i, v in enumerate(availabilities):
                    for j in DATES[index - DATEDATASTART].get_shifts():
                        if j.get_indicator() in v:
                            j.add_available_person(PERSONS[i])
                            PERSONS[i].bin_preference.append(1)
                        else:
                            PERSONS[i].bin_preference.append(0)

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

# toolbox = base.Toolbox()

if __name__ == "__main__":

    args = parser.parse_args()
    if args.generations:
        max_generations = int(args.generations)
    if args.input:
        file_name = args.input

    if os.path.isfile(file_name):
        read_availabilities(file_name)
        rrsp = RoomResponsibleSchedulingProblem()

        # # define a single objective, maximizing fitness strategy:
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        #
        # # create the Individual class based on list:
        # creator.create("Individual", list, fitness=creator.FitnessMin)
        #
        # # create an operator that randomly returns 0 or 1:
        # toolbox.register("zeroOrOne", random.randint, 0, 1)
        #
        # # create the individual operator to fill up an Individual instance:
        # toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(rrsp))
        #
        # # create the population operator to generate a list of individuals:
        # toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
        #
        #
        # # fitness calculation
        # def get_cost(individual):
        #     return rrsp.get_cost(individual),  # return a tuple
        #
        #
        # toolbox.register("evaluate", get_cost)
        #
        # # genetic operators:
        # toolbox.register("select", tools.selTournament, tournsize=2)
        # toolbox.register("mate", tools.cxTwoPoint)
        # toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(rrsp))
        #
        # # create initial population (generation 0):
        # population = toolbox.populationCreator(n=POPULATION_SIZE)
        #
        # # prepare the statistics object:
        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("min", numpy.min)
        # stats.register("avg", numpy.mean)
        #
        # # define the hall-of-fame object:
        # hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
        #
        # # perform the Genetic Algorithm flow with hof feature added:
        # population, logbook = ea_simple_with_elitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
        #                                              ngen=max_generations, stats=stats, halloffame=hof, verbose=True)
        #
        # # print best solution found:
        # best = hof.items[0]
        # print("-- Best Individual = ", best)
        # print("-- Best Fitness = ", best.fitness.values[0])
        # print()
        # print("-- Schedule = ")
        # rrsp.print_schedule_info(best)

        solve()

        for i in PERSONS:
            i.assign_from_bin()

        print_results()
