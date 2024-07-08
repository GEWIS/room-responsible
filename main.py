# Person class
class Person:
    def __init__(self, name, is_board):
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
        self.is_board = is_board
        self.max_shifts = 0
