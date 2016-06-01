# -*- coding: utf-8
import pytest
from emc import (Not, And, Or, AU, EU, AF, EF, AG, EG, P, Formula,
                 UnOp, BinOp, TRUE, FALSE)
from emc import TransitionGraph, Model


# See p. 248
mutualExclusionProblem = TransitionGraph(
    {
        (0, 'N1', 'N2'): {(1, 'T1', 'N2'), (2, 'N1', 'T2')},
        (1, 'T1', 'N2'): {(3, 'C1', 'N2'), (4, 'T1', 'T2')},
        (2, 'N1', 'T2'): {(5, 'T1', 'T2'), (6, 'N1', 'C2')},
        (3, 'C1', 'N2'): {(7, 'C1', 'T2'), (0, 'N1', 'N2')},
        (4, 'T1', 'T2'): {(7, 'C1', 'T2')},
        (5, 'T1', 'T2'): {(8, 'T1', 'C2')},
        (6, 'N1', 'C2'): {(8, 'T1', 'C2'), (0, 'N1', 'N2')},
        (7, 'C1', 'T2'): {(2, 'N1', 'T2')},
        (8, 'T1', 'C2'): {(1, 'T1', 'N2')},
    },
    initial=(0, 'N1', 'N2'))


@pytest.fixture
def mutexModel():
    return Model(mutualExclusionProblem)


def test_mutualExclusionProblem(mutexModel):
    """
    s₀ ⊨ AG(T₁ → AF(C₁)) (p. 251)

    ( Where → is rewritten as: ¬T₁ ∨ AF(C₁) )
    """
    formula = AG(Or(Not(P('T1')), AF(P('C1'))))
    assert mutexModel.satisfies(formula)


class TestSanity(object):

    @pytest.mark.parametrize(
        'ctlFormula',
        [Not(P('C1')),
         P('N1'),
         EF(P("C1")),
         EF(P("C2")),
         Not(AF(And(P("C1"), P("C2"))))])
    def test_mutexExpectations(self, ctlFormula, mutexModel):
        assert mutexModel.satisfies(ctlFormula)


class FakeFormula(object):

    def __init__(self, shouldMatch, subformulae):
        self._subformulae = subformulae
        self._shouldMatch = shouldMatch

    def match(self, state, graph, model):
        if self._shouldMatch:
            model.addLabel(state, self)

    def labels(self, model, state):
        return self._shouldMatch

    def subformulae(self):
        return iter(self._subformulae)


class TestModel(object):
    """
    Tests for Model.
    """
    @pytest.fixture
    def model(self):
        graph = TransitionGraph({('S1', 'S2'): {('S1', 'S2')}},
                                initial=('S1', 'S2'))
        return Model(graph)

    def test_resetLabels(self, model):
        model.addLabel(('S1', 'S2'), "some formula")
        model.resetLabels()
        assert not model.labeled(('S1', 'S2'), "some formula")

    def test_addLabel_labeled(self, model):
        assert not model.labeled(('S1', 'S2'), "some formula")
        model.addLabel(('S1', 'S2'), "some formula")
        assert model.labeled(('S1', 'S2'), "some formula")

    def test_orderSubformulae(self, model):
        """
        See (p. 251)
        """
        formula = AU(Not(P("X")), Or(P("Y"), P("Z")))
        ordered = list(model._orderSubformulae(formula))
        assert ordered == [P('Z'),
                           P('Y'),
                           P('X'),
                           Or(P('Y'),
                              P('Z')),
                           Not(P('X')),
                           AU(Not(P('X')),
                              Or(P('Y'),
                                 P('Z')))]

    def test_satisfies(self, model):
        subformula = FakeFormula(True, [])
        formula = FakeFormula(True, [subformula])
        assert model.satisfies(formula)
        assert model.labels == {('S1', 'S2'): {formula, subformula}}

        subformula = FakeFormula(True, [])
        formula = FakeFormula(False, [subformula])
        assert not model.satisfies(formula)
        assert model.labels == {('S1', 'S2'): {subformula}}


class TestTransitionGraph(object):
    initial = ('S1', 'S2')

    @pytest.fixture
    def graph(self):
        return TransitionGraph({('S1', 'S1'): {('S2', 'S2'), ('S3', 'S3')}},
                               initial=self.initial)

    def test_initial(self, graph):
        assert graph.initial == self.initial

    def test_successors(self, graph):
        assert graph.successors(('S1', 'S1')) == frozenset([
            ('S2', 'S2'),
            ('S3', 'S3')
        ])
        assert graph.successors(('S2', 'S2')) == frozenset([])

    def test_predecessors(self, graph):
        assert graph.predecessors(('S2', 'S2')) == frozenset([
            ('S1', 'S1'),
        ])
        assert graph.predecessors(('S1', 'S1')) == frozenset([])

    def test_states(self, graph):
        assert graph.states() == frozenset([('S1', 'S1'),
                                            ('S2', 'S2'),
                                            ('S3', 'S3')])

    def test_addEdge(self, graph):
        graph.addEdge(('S2', 'S2'), {('S4', 'S4')})
        assert graph.initial == self.initial
        assert ('S4', 'S4') in graph.states()
        assert graph.successors(('S2', 'S2')) == frozenset([('S4', 'S4')])
        assert graph.predecessors(('S4', 'S4')) == frozenset([('S2', 'S2')])


class FakeModel(object):

    def __init__(self):
        self.labels = {}

    def labeled(self, state, formula):
        return formula in self.labels.get(state, ())

    def addLabel(self, state, formula):
        self.labels.setdefault(state, []).append(formula)


@pytest.fixture
def model():
    return FakeModel()


class Subformula(Formula):
    matched = "matched"
    yielded = "subformula"

    def match(self, state, graph, model):
        return self.matched

    def subformulae(self):
        yield self.yielded


class TestFormula(object):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def formula(self):
        return Formula()

    def test_is_formula(self, formula):
        assert isinstance(formula, Formula)

    def test_identity(self, formula):
        assert formula.identity is formula

    def test_labels(self, model, formula):
        assert not formula.labels(model, 'some state')
        model.labels['some state'] = [formula]
        assert formula.labels(model, 'some state')

    def test_addLabel(self, model, formula):
        assert 'some state' not in model.labels
        formula.addLabel(model, 'some state')
        assert 'some state' in model.labels

    def test_match(self, graph, model, formula):
        assert not formula.match("some state", graph, model)

    def test_subformulae(self, formula):
        assert not list(formula.subformulae())

    def test_rewriteAs(self, graph, model, formula):
        subformula = Subformula()
        formula.rewriteAs(subformula)

        assert subformula.identity is formula
        assert formula.match('state',
                             graph,
                             model) is subformula.match('state',
                                                        graph,
                                                        model)
        assert list(formula.subformulae()) == list(subformula.subformulae())

    def _test_formula_given(self, formula, state, graph, model, expected):
        if expected:
            assert formula.match(state, graph, model)
            assert formula in model.labels[state]
        else:
            assert not formula.match(state, graph, model)
            assert formula not in model.labels.get(state, ())


class TestP(TestFormula):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def formula(self):
        return P(TRUE)

    def test_eq(self):
        assert P("A") == P("A")
        assert P("A") != P("B")
        assert P("A") != 123

    def test_hash(self):
        assert hash(P("A")) == hash(P("A"))
        assert hash(P("A")) != hash(P("B"))

    def test_repr(self):
        repr(P("A"))

    @pytest.mark.parametrize(
        "Pstates,state,expected",
        [(("A",), "A", True),
         (("A",), "B", False),
         (("A"), ("A", "B"), True),
         (("A", "B"), ("A", "B"), True),
         (("B", "A"), "C", False),
         ((TRUE,), "X", True),
         ((FALSE,), "X", False)])
    def test_match(self, Pstates, state, graph, model, expected):
        self._test_formula_given(P(*Pstates), state, graph, model, expected)


class TestUnOp(TestFormula):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def anUnOp(self):
        return UnOp(P(TRUE))

    @pytest.fixture
    def formula(self, anUnOp):
        return anUnOp

    def test_eq(self):
        assert UnOp("A") == UnOp("A")
        assert UnOp("A") != UnOp("B")
        assert UnOp("A") != 123

    def test_hash(self):
        assert hash(UnOp("A")) == hash(UnOp("A"))
        assert hash(UnOp("A")) != hash(UnOp("B"))

    def test_repr(self):
        repr(UnOp("A"))

    def test_arg1_required(self):
        with pytest.raises(TypeError):
            UnOp()
        UnOp("flop").arg1 == "flop"

    def test_subformulae(self, anUnOp):
        assert list(anUnOp.subformulae()) == [P(TRUE)]


class TestBinOp(TestFormula):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def aBinOp(self):
        return BinOp(P(TRUE), P(FALSE))

    @pytest.fixture
    def formula(self, aBinOp):
        return aBinOp

    def test_eq(self):
        assert BinOp("A", "B") == BinOp("A", "B")
        assert BinOp("A", "B") != BinOp("B", "A")
        assert BinOp("A", "B") != 123

    def test_hash(self):
        assert hash(BinOp("A", "B")) == hash(BinOp("A", "B"))
        assert hash(BinOp("A", "B")) != hash(BinOp("B", "A"))

    def test_repr(self):
        repr(BinOp("A", "B"))

    def test_arg1_required(self):
        with pytest.raises(TypeError):
            BinOp()
        with pytest.raises(TypeError):
            BinOp(arg2="X")

        BinOp("flip", "flop").arg1 == "flip"
        BinOp("flip", "flop").arg2 == "flop"

    def test_subformulae(self, aBinOp):
        assert list(aBinOp.subformulae()) == [P(TRUE), P(FALSE)]


class TestNot(TestUnOp):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def formula(self):
        return Not(P(TRUE))

    @pytest.mark.parametrize(
        'arg1,expected',
        [(P(TRUE), False),
         (P(FALSE), True)])
    def test_match(self, arg1, graph, model, expected):
        self._test_formula_given(Not(arg1),
                                 state="irrelevant",
                                 graph=graph,
                                 model=model,
                                 expected=expected)


class TestAnd(TestBinOp):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def formula(self):
        return And(P(TRUE), P(TRUE))

    @pytest.mark.parametrize(
        'arg1,arg2,expected',
        [(P(TRUE), P(TRUE), True),
         (P(TRUE), P(FALSE), False),
         (P(FALSE), P(TRUE), False),
         (P(FALSE), P(FALSE), False)])
    def test_match(self, arg1, arg2, graph, model, expected):
        self._test_formula_given(And(arg1, arg2),
                                 state='irrelevant',
                                 graph=graph, model=model, expected=expected)


class TestOr(TestBinOp):

    @pytest.fixture
    def graph(self):
        return None

    @pytest.fixture
    def formula(self):
        return Or(P(TRUE), P(TRUE))

    @pytest.mark.parametrize(
        'arg1,arg2,expected',
        [(P(TRUE), P(TRUE), True),
         (P(TRUE), P(FALSE), True),
         (P(FALSE), P(TRUE), True),
         (P(FALSE), P(FALSE), False)])
    def test_match(self, arg1, arg2, graph, model, expected):
        self._test_formula_given(Or(arg1, arg2),
                                 state='irrelevant',
                                 graph=graph, model=model, expected=expected)
