import concurrent.futures
from faker import Faker
from datetime import datetime, timedelta
import numpy as np
import os
from main import Person, Shift, Date, RoomResponsibleSchedulingProblem, ea_simple_with_elitism, read_availabilities
from deap import creator, base, tools

# Constants
MAX_GENERATIONS = 2000
NUM_TESTS = 10  # adjust as needed
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.2
HALL_OF_FAME_SIZE = 30
RANDOM_SEED = 42

fake = Faker(['en_US', 'nl_NL'])


def get_random_person():
    person = Person(fake.first_name())
    person.set_max_shifts(fake.random_int(min=-1, max=5))
    person.set_board(fake.boolean())
    person.set_bin_preference([fake.boolean() for _ in range(len(DATES) * len(SHIFTS))])
    return person


def generate_csv():
    print("Generating CSV file...")
    with open('test.csv', 'w', newline='') as file:
        for shift in SHIFTS:
            file.write(
                f'{datetime.strftime(shift.get_start_time(), "%H:%M:%S")};{datetime.strftime(shift.get_end_time(), "%H:%M:%S")};{shift.get_indicator()};')
        file.write('\n')
        file.write(f';monday;exam;{";".join([person.get_name() for person in PERSONS])}\n')
        file.write(f'max_shifts;;;{";".join((str(person.get_max_shifts()) for person in PERSONS))}\n')
        file.write(f'board;;;{";".join([str(int(person.get_is_board())) for person in PERSONS])}\n')
        for k, v in enumerate(DATES):
            availabilities = ''
            for i, person in enumerate(PERSONS):
                availability = ''
                for j, shift in enumerate(SHIFTS):
                    if person.get_bin_preference()[k * len(SHIFTS) + j]:
                        availability += f'{shift.get_indicator()}'
                availabilities += ';' + availability
            file.write(f'{datetime.strftime(v.get_date(), "%m/%d/%Y")};0;{int(v.get_date().weekday()==0)};{availabilities}\n')


def get_weekdays_in_current_month():
    now = datetime.now()
    first_day_of_month = now.replace(day=1)
    next_month = first_day_of_month.replace(day=28) + timedelta(days=4)  # this will never fail
    last_day_of_month = next_month - timedelta(days=next_month.day)
    weekdays = []
    current_date = first_day_of_month
    while current_date <= last_day_of_month:
        if current_date.weekday() < 5:  # 0: Monday, 1: Tuesday, ..., 4: Friday
            weekdays.append(current_date)
        current_date += timedelta(days=1)
    return weekdays


def run_evolution(dummy_arg):
    file_name = 'test.csv'
    if os.path.isfile(file_name):
        print(f"Reading availabilities from {file_name}...")
        read_availabilities(file_name)
        rrsp = RoomResponsibleSchedulingProblem()

        # Check if DEAP classes already exist and do not redefine them
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("zeroOrOne", fake.random_int, 0, 1)
        toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(rrsp))
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

        def get_cost(individual):
            return rrsp.get_cost(individual),

        toolbox.register("evaluate", get_cost)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(rrsp))

        population = toolbox.populationCreator(n=POPULATION_SIZE)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        print("Starting evolution...")
        population, logbook = ea_simple_with_elitism(population, toolbox, cxpb=P_CROSSOVER,
                                                     mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                                     stats=stats, halloffame=hof, verbose=True)

        for gen, record in enumerate(logbook):
            if record['min'] == 0.0:  # Assuming a fitness of 0 means optimality
                print(f"Optimal solution found in generation {gen + 1}")
                return gen + 1  # generations are 0-indexed, so add 1
        return MAX_GENERATIONS


if __name__ == "__main__":
    print("Setting up shifts and dates...")
    SHIFTS = [
        Shift("8:30:00", "13:00:00", "M"),
        Shift("13:00:00", "17:00:00", "A"),
    ]
    DATES = [Date(False, 0, i) for i in get_weekdays_in_current_month()]
    PERSONS = [get_random_person() for _ in range(15)]

    for shift in SHIFTS:
        for date in DATES:
            date.add_shift(shift)

    generate_csv()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        print(f"Running {NUM_TESTS} tests in parallel...")
        results = list(executor.map(run_evolution, range(NUM_TESTS)))

    average_generations = sum(results) / len(results)

    print(f"Average generations needed: {average_generations}")
    print(f"Results: {results}")