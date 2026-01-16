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
import pulp

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
        self.weight = int(weight)

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


def solvepulp():
    global BIN_WEIGHTS
    SHIFTSTOT = len(SHIFTS) * len(DATES)

    N = [] # People with max shifts assigned
    B = [] # Board members
    BI = [] # Board members with infinite shifts

    ma = pulp.LpProblem(name="rrc", sense=pulp.LpMaximize)

    for j, person in enumerate(PERSONS):
        if person.get_is_board():
            B.append(j)
            if person.get_max_shifts() == -1:
                BI.append(j)
        if person.get_max_shifts() != -1:
            N.append(j)


    # Variables
    r = [pulp.LpVariable(name=f"r_{i}", cat=pulp.LpInteger, lowBound=0) for i in range(SHIFTSTOT)]
    b = [pulp.LpVariable(name=f"b_{i}", cat=pulp.LpInteger, lowBound=0) for i in range(SHIFTSTOT)]
    x = [[pulp.LpVariable(name=f"x_{i}_{j}", cat=pulp.LpBinary, lowBound=0, upBound=1) for j in range(len(PERSONS))] for i in range(SHIFTSTOT)]
    l = [[pulp.LpVariable(name=f"l_{i}_{j}", cat=pulp.LpInteger, lowBound=0) for j in range(len(PERSONS) + 1)] for i in range(SHIFTSTOT//3)]
    n = [pulp.LpVariable(name=f"n_{i}", cat=pulp.LpInteger, lowBound=0) for i in range(len(N))]
    bv = [pulp.LpVariable(name=f"bv_{i}", cat=pulp.LpInteger, lowBound=0) for i in range(len(B))]
    mean = pulp.LpVariable(name="mean", cat=pulp.LpInteger, lowBound=0)

    # Availability constraint
    for i in range(SHIFTSTOT):
        for j, person in enumerate(PERSONS):
            ma += (x[i][j] <= person.get_bin_preference()[i], f"available_{i}_{j}")


    for i in range(SHIFTSTOT):
        # Extra variable for people assigned to shift
        ma += (pulp.lpSum(x[i]) == r[i], f"rge_{i}")
        ma += (r[i] <= 2, f"ass_1_{i}")

        # All shifts have at least one board member
        sum = pulp.lpSum([x[i][j] for j in B])
        ma += (sum >= b[i], f"boardav_{i}")

    for i in range(0, SHIFTSTOT, 3):
        for j in range(len(PERSONS)):
            xm = pulp.lpSum([x[i][j], x[i+1][j], x[i+1][j], x[i+2][j]])
            a = 3
            A = 4
            ma += (0 <= xm, f"l_{i//3}_{j}_1")
            ma += (xm <= a * l[i//3][j], f"l_{i//3}_{j}_2")
            ma += (a + A * (l[i//3][j] - 1) <= xm, f"l_{i//3}_{j}_3") 
            ma += (xm <= a + A * (l[i//3][j]) - 1, f"l_{i//3}_{j}_4") 
        ma += (l[i//3][-1] == pulp.lpSum(l[i//3][:-1]), f"l_{i//3}")
         
    # People with a max shifts get maximum their max shifts. 
    for i, j in enumerate(N):
        person = PERSONS[j]
        c = pulp.LpAffineExpression([(expr, BIN_WEIGHTS[i]) for i, expr in enumerate(get_column(x, j))])
        ma += (c == n[i], f"n_{j}")
        # ma += (pulp.lpSum([BIN_WEIGHTS[i] * expr for i, expr in enumerate(get_column(x, j))] == n[i], f"bv_{j}"))
        ma += (n[i] <= person.get_max_shifts(), f"maxshift_{j}")

    for i, j in enumerate(B):
        c = pulp.LpAffineExpression([(expr, BIN_WEIGHTS[i]) for i, expr in enumerate(get_column(x, j))])
        ma += (c == bv[i], f"bv_{j}")
        # ma += (pulp.lpSum([BIN_WEIGHTS[i] * expr for i, expr in enumerate(get_column(x, j))] == bv[i], f"bv_{j}"))

    # ma += (mean == (1/len(B)) * pulp.lpSum(bv[i] for i in B))
    #
    # variance = (1/len(B)) * pulp.lpSum((bv[i] - mean)*(bv[i] - mean) for i in B)

    ma += 5 * pulp.lpSum(r) + pulp.lpSum(b) + pulp.lpSum(n)# + pulp.lpSum([l[i//3][-1] for i in range(0, SHIFTSTOT, 3)])# - variance

    ma.writeLP("NameofFile.lp")
    ma.solve()

    print(f"status: {ma.status}, {pulp.LpStatus[ma.status]}")
    print(f"objective: {ma.objective.value()}")

    # for name, constraint in ma.constraints.items():
    #     print(f"{name}: {constraint.value()}")
    
    for v in get_column(l, -1):
        try: 
            print('%s %g' % (v.name, v.value()))
        except:
            print('%s' % (v.name))
    
    bin_prefs = [[0 for _ in range(SHIFTSTOT)] for _ in range(len(PERSONS))]
    for v in flatten(x):
        index = re.split(r'_', v.name)
        if (index[0] == "x"):
            shift, person = int(index[1]), int(index[2])
            bin_prefs[person][shift] = int(v.value())
    #
    for i in range(len(PERSONS)):
        PERSONS[i].set_bin_assignment(bin_prefs[i])
        print(PERSONS[i].get_bin_assignment())
        print(len([x for i, x in enumerate(PERSONS[i].get_bin_assignment()) if i % 3 == 1]))


def flatten(xss):
    return [x for xs in xss for x in xs]

from pyscipopt import Model, quicksum, recipes
from pyscipopt.recipes import nonlinear
def solvescip():
    global BIN_WEIGHTS
    SHIFTSTOT = len(SHIFTS) * len(DATES)

    N = [] # People with max shifts assigned
    B = [] # Board members
    BI = [] # Board members with infinite shifts

    m = Model()

    for j, person in enumerate(PERSONS):
        if person.get_is_board():
            B.append(j)
            if person.get_max_shifts() == -1:
                BI.append(j)
        if person.get_max_shifts() != -1:
            N.append(j)


    # Variables
    r = [m.addVar(vtype="I", name=f"r_{i}") for i in range(SHIFTSTOT)]
    b = [m.addVar(vtype="I", name=f"b_{i}") for i in range(SHIFTSTOT)]
    x = [[m.addVar(vtype="B", name=f"x_{i}_{j}") for j in range(len(PERSONS))] for i in range(SHIFTSTOT)]
    # x = m.addMVar(shape=(SHIFTSTOT, len(PERSONS)), vtype="B", name="x")
    l = [m.addVar(vtype="I", name=f"l_{i}") for i in range(SHIFTSTOT//3)]
    n = [m.addVar(vtype="I", name=f"n_{i}") for i in range(len(N))]
    bv = [m.addVar(vtype="I", name=f"bv_{i}") for i in range(len(B))]
    mean = m.addVar(lb=-GRB.INFINITY, name="mean")

    # Availability constraint
    for i in range(SHIFTSTOT):
        for j, person in enumerate(PERSONS):
            m.addCons(x[i][j] <= person.get_bin_preference()[i], f"available_{i}_{j}")


    for i in range(SHIFTSTOT):
        # Extra variable for people assigned to shift
        m.addCons(quicksum(x[i]) == r[i], f"rge_{i}")
        m.addCons(r[i] <= 2, f"ass_1_{i}")

        # All shifts have at least one board member
        sum = quicksum([x[i][j] for j in B])
        m.addCons(sum >= b[i], f"boardav_{i}")

    for i in range(0, SHIFTSTOT, 3):
        l1 = rowmult(x[i], x[i+1])
        l2 = rowmult(x[i+2], x[i+1])
        l3 = rowmult(l1, l2)
        l4 = quicksum(l1) + quicksum(l2) - quicksum(l3)
        m.addCons(l[i // 3] == l4, f"l_{i//3}")
        # m.addConstr(l[i // 3] <= 1, f"ltop_{i//3}")
         
    # People with a max shifts get maximum their max shifts. 
    for i, j in enumerate(N):
        person = PERSONS[j]
        m.addCons(wegrsum(get_column(x, j), BIN_WEIGHTS) == n[i], f"n_{j}")
        m.addCons(n[i] <= person.get_max_shifts(), f"maxshift_{j}")

    for i, j in enumerate(B):
        m.addCons(wegrsum(get_column(x, j), BIN_WEIGHTS) == bv[i], f"bv_{j}")

    # Constraint for mean
    m.addCons(mean == (1/len(B)) * quicksum(bv[i] for i in B))

    # Variance expression
    variance = (1/len(B)) * quicksum((bv[i] - mean)*(bv[i] - mean) for i in B)
    # Objective: minimize variance

    # m.setObjective(5 * quicksum(r) + quicksum(b) + quicksum(n) + quicksum(l) - variance, sense='maximize')
    nonlinear.set_nonlinear_objective(m, 5 * quicksum(r) + quicksum(b) + quicksum(n) + quicksum(l) - variance, sense='maximize')

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
            print('%s %g' % (v, m.getVal(v)))
        except:
            print('%s' % (v))

    bin_prefs = [[0 for _ in range(SHIFTSTOT)] for _ in range(len(PERSONS))]
    for v in m.getVars():
        index = re.split(r'_', str(v))
        if (index[0] == "x"):
            shift, person = int(index[1]), int(index[2])
            bin_prefs[person][shift] = int(m.getVal(v))

    for i in range(len(PERSONS)):
        PERSONS[i].set_bin_assignment(bin_prefs[i])
        print(PERSONS[i].get_bin_assignment())
        print(len([x for i, x in enumerate(PERSONS[i].get_bin_assignment()) if i % 3 == 1]))

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
    mean = m.addVar(lb=-GRB.INFINITY, name="mean")

    # Availability constraint
    for i in range(SHIFTSTOT):
        for j, person in enumerate(PERSONS):
            m.addConstr(x[i][j] <= person.get_bin_preference()[i], f"available_{i}_{j}")


    for i in range(SHIFTSTOT):
        # Extra variable for people assigned to shift
        m.addConstr(gp.quicksum(x[i]) == r[i], f"rge_{i}")
        m.addConstr(r[i] <= 2, f"ass_1_{i}")

        # All shifts have at least one board member
        sum = gp.LinExpr()
        for j in B:
            sum += x[i][j]
        m.addConstr(sum >= b[i], f"boardav_{i}")

    for i in range(0, SHIFTSTOT, 3):
        l1 = rowmult(x[i], x[i+1])
        l2 = rowmult(x[i+2], x[i+1])
        l3 = rowmult(l1, l2)
        l4 = gp.quicksum(l1) + gp.quicksum(l2) - gp.quicksum(l3)
        m.addConstr(l[i // 3] == l4, f"l_{i//3}")
        # m.addConstr(l[i // 3] <= 1, f"ltop_{i//3}")
         
    # People with a max shifts get maximum their max shifts. 
    for i, j in enumerate(N):
        person = PERSONS[j]
        m.addConstr(wegrsum(get_column(x, j), BIN_WEIGHTS) == n[i], f"n_{j}")
        m.addConstr(n[i] <= person.get_max_shifts(), f"maxshift_{j}")

    for i, j in enumerate(B):
        m.addConstr(wegrsum(get_column(x, j), BIN_WEIGHTS) == bv[i], f"bv_{j}")

    # Constraint for mean
    m.addConstr(mean == (1/len(B)) * gp.quicksum(bv[i] for i in B))

    # Variance expression
    variance = (1/len(B)) * gp.quicksum((bv[i] - mean)*(bv[i] - mean) for i in B)
    # Objective: minimize variance
    m.ModelSense = GRB.MAXIMIZE

    m.setObjective(5 * gp.quicksum(r) + gp.quicksum(b) + gp.quicksum(n) + gp.quicksum(l) - variance, GRB.MAXIMIZE)

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

# def grsum(x):
#     obj = gp.LinExpr()
#     for expr in x:
#         obj += expr
#     return obj

def rowmult(x1, x2):
    # obj = gp.LinExpr()
    obj = []
    for i, j in zip(x1, x2):
        obj.append(i * j)
    return obj

def wegrsum(x, weights):
    obj = 0
    for i, expr in enumerate(x):
        obj += weights[i] * expr
    return obj

def get_column(x, i) -> list:
    return [row[i] for row in x]


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

parser = argparse.ArgumentParser(description="List of arguments")
parser.add_argument("-i", "--input", help="Input file path")
# set the random seed:

if __name__ == "__main__":

    args = parser.parse_args()
    if args.input:
        file_name = args.input

    if os.path.isfile(file_name):
        read_availabilities(file_name)
        rrsp = RoomResponsibleSchedulingProblem()

        # solve()
        # solvepulp()
        solvescip()

        for i in PERSONS:
            i.assign_from_bin()

        print_results()
