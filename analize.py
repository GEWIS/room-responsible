from faker import Faker
from main import Person, Shift, Date
from datetime import datetime, timedelta

def get_random_person():
    person = Person(fake.first_name())
    person.set_max_shifts(fake.random_int(min=-1, max=5))
    person.set_board(fake.boolean())
    person.set_bin_preference([fake.boolean() for _ in range(len(DATES) * len(SHIFTS))])

    return person

def generate_csv():
    with open('test.csv', 'w', newline='') as file:
        for shift in SHIFTS:
            file.write(f'{datetime.strftime(shift.get_start_time(), "%H:%M:%S")};{datetime.strftime(shift.get_end_time(), "%H:%M:%S")};{shift.get_indicator()};')
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
            file.write(f'{datetime.strftime(v.get_date(), "%d/%m/%Y")};;{availabilities}\n')
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

fake = Faker(['en_US', 'nl_NL'])
SHIFTS = [
    Shift("8:30:00", "13:00:00", "M"),
    Shift("13:00:00", "17:00:00", "A"),
]
DATES = [Date(False, 0, i) for i in get_weekdays_in_current_month()]
PERSONS = [get_random_person() for _ in range(15)]

if __name__ == '__main__':
    for i in SHIFTS:
        for j in DATES:
            j.add_shift(i)
    generate_csv()