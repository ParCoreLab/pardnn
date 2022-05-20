# node class: name and level:
class NodeProps:
    def __init__(self, duration = 1, level = -1, start_time = -1 , end_time = -1, part = -1, parents = [], children = []):
        self.duration = duration
        self.level = level
        self.start_time = start_time
        self.end_time = end_time
        self.part = part
        self.parents = []
        self.children = []

    def representation(self):
        return "[ duration: " + str(self.duration) + ', level: ' + str(self.level)+ ', start time: ' + str(self.start_time) +  ', end_time: ' + str(self.end_time) + "]"




    """ def __eq__(self, other):
        return self.name == other

    

    def __lt__(self, other):
        return self.name < other

    def __gt__(self, other):
        return self.name > other
 """