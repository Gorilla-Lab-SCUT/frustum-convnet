from collections import OrderedDict
from utils.utils import AverageMeter


class TrainingStates(object):

    def __init__(self, state_names=[]):

        self.states = OrderedDict(
            [(key, 0) for key in state_names]
        )

        self.average_meters = OrderedDict(
            [(key, AverageMeter())
             for key in state_names]
        )

        self.state_names = state_names

    def update_states(self, states, batch_size=1):

        if len(self.states) == 0:
            state_names = states.keys()
            self.states = OrderedDict(
                [(key, 0) for key in state_names]
            )

            self.average_meters = OrderedDict(
                [(key, AverageMeter())
                 for key in state_names]
            )

        self.states.update(states)
        for key, meter in self.average_meters.items():
            meter.update(self.states[key], batch_size)

    def get_states(self, avg=True):
        states = OrderedDict()
        for key, meter in self.average_meters.items():
            if avg:
                states[key] = meter.avg
            else:
                states[key] = meter.val

        return states

    def format_states(self, states):
        output = ''
        for key, state in states.items():
            output += '%s: %.3f ' % (key, state)
        return output
