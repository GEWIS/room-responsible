from faker import Faker
from main import Person, Shift
from datetime import datetime

def get_random_person():
    person = Person(fake.first_name())
    person.set_max_shifts(fake.random_int(min=-1, max=5))
    person.set_board(fake.boolean())
    return person

def generate_csv():
    with open('test.csv', 'w', newline='') as file:
        for shift in SHIFTS:
            file.write(f'{datetime.strftime(shift.get_start_time(), "%H:%M:%S")};{datetime.strftime(shift.get_end_time(), "%H:%M:%S")};{shift.get_indicator()};')
        file.write('\n')
        file.write(f';monday;exam;{";".join([person.get_name() for person in PERSONS])}\n')
        file.write('max_shifts;;;;;;;;;;;;;\n')
        file.write(f'board;;;{";".join([str(int(person.get_is_board())) for person in PERSONS])}\n')

fake = Faker(['en_US', 'nl_NL'])
SHIFTS = [
    Shift("8:30:00", "13:00:00", "M"),
    Shift("13:00:00", "17:00:00", "A"),
]

PERSONS = [get_random_person() for _ in range(15)]

if __name__ == '__main__':
    generate_csv()