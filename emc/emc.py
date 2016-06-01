# -*- coding: utf-8
"""An implementation of the *unfair* Extended Model Checker described
in:

E. M. Clarke, E. A. Emerson, and A. P. Sistla. 1986.  Automatic
verification of finite-state concurrent systems using temporal logic
specifications. ACM Trans. Program. Lang. Syst. 8, 2 (April 1986),
244-263. DOI=http://dx.doi.org/10.1145/5397.5399

"""
from collections import deque

TRUE = 'TRUE'
FALSE = 'FALSE'


class Model(object):
    """
    A model instance labels states.
    """

    def __init__(self, graph):
        self.graph = graph
        self.resetLabels()

    def resetLabels(self):
        self.labels = {state: set() for state in self.graph.states()}

    def addLabel(self, state, formula):
        self.labels[state].add(formula)

    def labeled(self, state, formula):
        return formula in self.labels[state]

    def _orderSubformulae(self, formula):
        queue = deque([formula])
        toMatch = deque()

        while queue:
            subformula = queue.popleft()
            toMatch.appendleft(subformula)
            queue.extend(subformula.subformulae())

        return toMatch

    def satisfies(self, formula):
        self.resetLabels()
        for f in self._orderSubformulae(formula):
            for state in self.labels:
                f.match(state, self.graph, self)

        return formula.labels(model=self, state=self.graph.initial)


class TransitionGraph(object):
    """
    A directed graph that can tracks predecessors and successors of
    each node.
    """

    def __init__(self, adjacency, initial):
        self._successors = {}
        self._predecessors = {}
        self.initial = initial

        for source, targets in adjacency.items():
            self.addEdge(source, targets)

    def addEdge(self, source, targets):
        for target in targets:
            self._predecessors.setdefault(target, set()).add(source)
        self._successors.setdefault(source, set()).update(targets)

    def successors(self, node):
        return frozenset(self._successors.get(node, ()))

    def predecessors(self, node):
        return frozenset(self._predecessors.get(node, ()))

    def states(self):
        return frozenset(self._successors) | frozenset(self._predecessors)


class Formula(object):
    """
    A formula element.
    """

    def __init__(self, *args, **kwargs):
        super(Formula, self).__init__(*args, **kwargs)
        self.identity = self

    def labels(self, model, state):
        return model.labeled(state, self.identity)

    def addLabel(self, model, state):
        model.addLabel(state, self.identity)

    def rewriteAs(self, rule):
        """
        Rewrite self as rule.
        """
        rule.identity = self
        self.match = rule.match
        self.subformulae = rule.subformulae

    def match(self, state, graph, model):
        return False

    def subformulae(self):
        return iter([])


class P(Formula):
    """An atomic proposition.

    Currently only allows state assertions -- i.e., that a given state
    in the global transition graph contains some subset of states.
    """

    def __init__(self, *states):
        super(P, self).__init__()
        if states[0] is TRUE:
            self.states = TRUE
        else:
            self.states = frozenset(states)

    def match(self, state, graph, model):
        if self.states is FALSE:
            return False

        if self.states is TRUE or self.states.issubset(frozenset(state)):
            self.addLabel(model, state)
            return True

        return False

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.states == other.states

    def __hash__(self):
        return hash((self.__class__.__name__, self.states))

    def __repr__(self):
        cn = self.__class__.__name__
        if self.states in (TRUE, FALSE):
            states = self.states
        else:
            states = ', '.join(repr(state) for state in self.states)
        return '{}({})'.format(cn, states)


class UnOp(Formula):

    def __init__(self, arg1, *args, **kwargs):
        super(UnOp, self).__init__(*args, **kwargs)
        self.arg1 = arg1

    def __hash__(self):
        return hash((self.__class__.__name__, self.arg1))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.arg1 == other.arg1

    def __repr__(self):
        cn = self.__class__.__name__
        return '{}({!r})'.format(cn, self.arg1)

    def subformulae(self):
        yield self.arg1


class BinOp(Formula):

    def __init__(self, arg1, arg2, *args, **kwargs):
        super(BinOp, self).__init__(*args, **kwargs)
        self.arg1 = arg1
        self.arg2 = arg2

    def __hash__(self):
        return hash((self.__class__.__name__, self.arg1, self.arg2))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.arg1, self.arg2) == (other.arg1, other.arg2)

    def __repr__(self):
        cn = self.__class__.__name__
        return '{}({!r}, {!r})'.format(cn, self.arg1, self.arg2)

    def subformulae(self):
        yield self.arg1
        yield self.arg2


class Not(UnOp):
    """
    ¬arg1
    """

    def match(self, state, graph, model):
        if not self.arg1.match(state, graph, model):
            self.addLabel(model, state)
            return True
        return False


class And(BinOp):
    """
    arg1 ∧ arg2
    """

    def match(self, state, graph, model):
        if (self.arg1.match(state, graph, model) and
            self.arg2.match(state, graph, model)):
            self.addLabel(model, state)
            return True
        return False


class Or(BinOp):
    """
    arg1 ∨ arg2
    """

    def match(self, state, graph, model):
        if (self.arg1.match(state, graph, model) or
            self.arg2.match(state, graph, model)):
            self.addLabel(model, state)
            return True
        return False


class AU(BinOp):
    """
    A[arg1 U arg2]
    """

    def _match(self, state, graph, model, marked):
        if state in marked:
            return self.labels(model, state)

        marked.add(state)

        if self.arg2.labels(model, state):
            self.addLabel(model, state)
            return True

        if not self.arg1.labels(model, state):
            return False

        successors = graph.successors(state)
        if not successors:
            return False

        return all(self._match(successor, graph, model, marked)
                   for successor in successors)

    def match(self, state, graph, model):
        return self._match(state, graph, model, marked=set())


class EU(BinOp):
    """
    E[arg1 U arg2]
    """

    def _markArg1(self, state, graph, model, marked):
        if state in marked:
            return self.labels(model, state)

        marked.add(state)

        if not self.arg1.labels(model, state):
            return False

        self.addLabel(model, state)

        predecessors = graph.predecessors(state)
        if not predecessors:
            return True

        return any(self._markArg1(predecessor, graph, model, marked)
                   for predecessor in predecessors)

    def _match(self, state, graph, model, marked):
        if state in marked:
            return self.labels(model, state)
        marked.add(state)
        if self.arg2.labels(model, state):
            # search backwards
            return any(self._markArg1(predecessor, graph, model, marked)
                       for predecessor in graph.predecessors(state))

        return any(self._match(state, graph, model, marked)
                   for successor in graph.successors(state))

    def match(self, state, graph, model):
        marked = set()
        return self._match(state, graph, model, marked)


class AF(UnOp):
    """AF(arg1) ≡ A[True U arg1]

    ...intuitively means that arg1 holds in the future along every
    path from the transition graph's initial state; that is, arg1 is
    inevitable.

    """

    def __init__(self, arg1):
        super(AF, self).__init__(arg1)
        self.rewriteAs(AU(P(TRUE), arg1))


class EF(UnOp):
    """EF(arg1) ≡ E[True U arg1]

    ...means that there is some path from the transition graph's
    initial state that leads to a state at which arg1 holds; that is,
    arg1 potentially holds.

    """

    def __init__(self, arg1):
        super(EF, self).__init__(arg1)
        self.rewriteAs(EU(P(TRUE), arg1))


class EG(UnOp):
    """EG(arg1) ≡ ¬AF(¬arg1)

    ...means that there is some path from the transition graph's
    initial state on which f holds at every state.

    """
    def __init__(self, arg1):
        super(EG, self).__init__(arg1)
        self.rewriteAs(Not(AF(Not(arg1))))


class AG(UnOp):
    """AG(arg1) ≡ ¬EF(¬arg1)

    ...means that arg1 holds at every state on every path from the the
    transition graph's initial state; that is, f holds globally.

    """

    def __init__(self, arg1):
        super(AG, self).__init__(arg1)
        self.rewriteAs(Not(EF(Not(arg1))))
