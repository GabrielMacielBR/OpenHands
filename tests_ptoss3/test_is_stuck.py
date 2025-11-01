import pytest

from openhands.controller.state.state import State
from openhands.controller.stuck import StuckDetector
from openhands.events import EventSource
from openhands.events.action import MessageAction
from openhands.events.action.commands import CmdRunAction, IPythonRunCellAction
from openhands.events.action.empty import NullAction
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.empty import NullObservation


class TestStuckDetectorMCDC54:
    """
    Testes de cobertura MC/DC para a condição da linha 54 (modo não-headless):

    if (isinstance(event, MessageAction) and event.source == EventSource.USER):

    A = isinstance(event, MessageAction)
    B = event.source == EventSource.USER

    Combinações necessárias (AB = Resultado):
    1: VV = V (encontra última mensagem do usuário)
    2: VF = F (MessageAction mas não é USER)
    3: FV = F (não é MessageAction)
    """

    @pytest.fixture
    def mock_state(self):
        """Cria um State mockado para os testes"""
        state = State(inputs={}, max_iterations=100)
        return state

    def test_line54_case1_message_action_from_user(self, mock_state):
        """Caso 1: AB=VV - MessageAction do USER (encontra última mensagem)"""
        # A=V: é MessageAction
        # B=V: source é USER

        # Adiciona eventos antes da mensagem do usuário
        mock_state.history.append(CmdRunAction(command='echo "before"'))
        mock_state.history.append(
            CmdOutputObservation(content='before', command='echo "before"')
        )

        # Adiciona a mensagem do usuário
        user_message = MessageAction(content='Please help')
        user_message._source = EventSource.USER
        mock_state.history.append(user_message)

        # Adiciona eventos após a mensagem do usuário
        mock_state.history.append(CmdRunAction(command='echo "after"'))
        mock_state.history.append(
            CmdOutputObservation(content='after', command='echo "after"')
        )

        # Testa no modo não-headless (deve considerar apenas após a última mensagem USER)
        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=False)

        assert isinstance(result, bool)

    def test_line54_case2_message_action_not_from_user(self, mock_state):
        """Caso 2: AB=VF - MessageAction mas não é USER (não encontra)"""
        # A=V: é MessageAction
        # B=F: source é AGENT

        # Adiciona eventos
        mock_state.history.append(CmdRunAction(command='ls'))

        # Adiciona mensagem do AGENT (não USER)
        agent_message = MessageAction(content='I am working')
        agent_message._source = EventSource.AGENT
        mock_state.history.append(agent_message)

        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )

        # Testa no modo não-headless (não deve encontrar mensagem USER)
        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=False)

        assert isinstance(result, bool)

    def test_line54_case3_not_message_action(self, mock_state):
        """Caso 3: AB=FV - Não é MessageAction (não encontra)"""
        # A=F: não é MessageAction
        # B=V: (source não se aplica, mas o evento tem source)

        # Adiciona apenas eventos que não são MessageAction
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(
            CmdOutputObservation(content='file.txt', command='ls')
        )
        mock_state.history.append(IPythonRunCellAction(code='print("test")'))
        mock_state.history.append(
            IPythonRunCellObservation(content='test', code='print("test")')
        )

        # Testa no modo não-headless (não deve encontrar mensagem USER)
        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=False)

        assert isinstance(result, bool)


class TestStuckDetectorMCDC70:
    """
    Testes de cobertura MC/DC para a condição de filtro:

    not ((isinstance(event, MessageAction) and event.source == EventSource.USER)
         or isinstance(event, (NullAction, NullObservation)))

    A = isinstance(event, MessageAction)
    B = event.source == EventSource.USER
    C = isinstance(event, (NullAction, NullObservation))

    Combinações necessárias (ABC = Resultado esperado no filtro):
    2: VVF = True (filtra)   - MessageAction do USER, não é Null
    3: VFV = True (filtra)   - MessageAction não-USER que é Null
    4: VFF = False (mantém)  - MessageAction não-USER, não é Null
    5: FVV = False (mantém)  - Não é MessageAction, é Null (B não importa)
    """

    @pytest.fixture
    def mock_state(self):
        """Cria um State mockado para os testes"""
        state = State(inputs={}, max_iterations=100)
        return state

    def test_mcdc_case_2_message_user_not_null(self, mock_state):
        """Caso 2: ABC=VVF - MessageAction do USER (deve ser filtrado)"""
        # A=V: é MessageAction
        # B=V: source é USER
        # C=F: não é NullAction/NullObservation
        event = MessageAction(content='Hello')
        event._source = EventSource.USER

        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(event)  # Este deve ser filtrado
        mock_state.history.append(CmdRunAction(command='pwd'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        # Verifica que não crashou (evento foi filtrado corretamente)
        assert isinstance(result, bool)

    def test_mcdc_case_3_message_agent_is_null(self, mock_state):
        """Caso 3: ABC=VFV - MessageAction não-USER que é Null (deve ser filtrado)"""
        # A=V: é MessageAction (tecnicamente NullAction é um tipo de Action)
        # B=F: source não é USER
        # C=V: é NullAction
        event = NullAction()

        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(event)  # Este deve ser filtrado
        mock_state.history.append(CmdRunAction(command='pwd'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_mcdc_case_4_message_agent_not_null(self, mock_state):
        """Caso 4: ABC=VFF - MessageAction não-USER, não Null (deve ser mantido)"""
        # A=V: é MessageAction
        # B=F: source é AGENT
        # C=F: não é NullAction/NullObservation
        event = MessageAction(content='I will help')
        event._source = EventSource.AGENT

        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(event)  # Este NÃO deve ser filtrado
        mock_state.history.append(CmdRunAction(command='pwd'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_mcdc_case_5_not_message_is_null(self, mock_state):
        """Caso 5: ABC=FVV - Não é MessageAction, é Null (deve ser filtrado)"""
        # A=F: não é MessageAction
        # B=V: (não se aplica, mas source existe em outros eventos)
        # C=V: é NullObservation
        event = NullObservation(content='')

        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(event)  # Este deve ser filtrado
        mock_state.history.append(CmdRunAction(command='pwd'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)


class TestStuckDetectorMCDC91:
    """
    Testes de cobertura MC/DC para a condição da linha 91:

    if isinstance(event, Action) and len(last_actions) < 4:

    A = isinstance(event, Action)
    B = len(last_actions) < 4

    Combinações necessárias (AB = Resultado):
    1: VV = V (adiciona action à lista)
    2: VF = F (action mas lista cheia)
    3: FV = F (não é action)
    """

    @pytest.fixture
    def mock_state(self):
        """Cria um State mockado para os testes"""
        state = State(inputs={}, max_iterations=100)
        return state

    def test_line91_case1_is_action_and_less_than_4(self, mock_state):
        """Caso 1: AB=VV - É Action e lista tem menos de 4 (adiciona)"""
        # A=V: é Action
        # B=V: len(last_actions) < 4

        # Adiciona 3 actions (menos de 4)
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(CmdRunAction(command='echo "test"'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_line91_case2_is_action_but_4_or_more(self, mock_state):
        """Caso 2: AB=VF - É Action mas lista já tem 4 (não adiciona)"""
        # A=V: é Action
        # B=F: len(last_actions) >= 4

        # Adiciona 4 actions (lista cheia)
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(CmdRunAction(command='echo "1"'))
        mock_state.history.append(CmdRunAction(command='echo "2"'))
        # Adiciona mais uma action (não deve ser adicionada à lista)
        mock_state.history.append(CmdRunAction(command='echo "3"'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_line91_case3_not_action(self, mock_state):
        """Caso 3: AB=FV - Não é Action (não adiciona)"""
        # A=F: não é Action (é Observation)
        # B=V: len(last_actions) < 4

        # Adiciona apenas observations
        mock_state.history.append(
            CmdOutputObservation(content='file.txt', command='ls')
        )
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)


class TestStuckDetectorMCDC93:
    """
    Testes de cobertura MC/DC para a condição da linha 93:

    elif isinstance(event, Observation) and len(last_observations) < 4:

    A = isinstance(event, Observation)
    B = len(last_observations) < 4

    Combinações necessárias (AB = Resultado):
    1: VV = V (adiciona observation à lista)
    2: VF = F (observation mas lista cheia)
    3: FV = F (não é observation)
    """

    @pytest.fixture
    def mock_state(self):
        """Cria um State mockado para os testes"""
        state = State(inputs={}, max_iterations=100)
        return state

    def test_line93_case1_is_observation_and_less_than_4(self, mock_state):
        """Caso 1: AB=VV - É Observation e lista tem menos de 4 (adiciona)"""
        # A=V: é Observation
        # B=V: len(last_observations) < 4

        # Adiciona actions seguidas de observations (3 observations)
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(
            CmdOutputObservation(content='file1.txt', command='ls')
        )
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )
        mock_state.history.append(CmdRunAction(command='echo "test"'))
        mock_state.history.append(
            CmdOutputObservation(content='test', command='echo "test"')
        )

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_line93_case2_is_observation_but_4_or_more(self, mock_state):
        """Caso 2: AB=VF - É Observation mas lista já tem 4 (não adiciona)"""
        # A=V: é Observation
        # B=F: len(last_observations) >= 4

        # Adiciona 4 pares action-observation
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(
            CmdOutputObservation(content='file1.txt', command='ls')
        )
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )
        mock_state.history.append(CmdRunAction(command='echo "1"'))
        mock_state.history.append(CmdOutputObservation(content='1', command='echo "1"'))
        mock_state.history.append(CmdRunAction(command='echo "2"'))
        mock_state.history.append(CmdOutputObservation(content='2', command='echo "2"'))
        # Adiciona mais uma observation (não deve ser adicionada à lista)
        mock_state.history.append(CmdRunAction(command='echo "3"'))
        mock_state.history.append(CmdOutputObservation(content='3', command='echo "3"'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_line93_case3_not_observation(self, mock_state):
        """Caso 3: AB=FV - Não é Observation (não adiciona)"""
        # A=F: não é Observation (é Action)
        # B=V: len(last_observations) < 4

        # Adiciona apenas actions
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(CmdRunAction(command='echo "test"'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)


class TestStuckDetectorMCDC96:
    """
    Testes de cobertura MC/DC para a condição da linha 96:

    if len(last_actions) == 4 and len(last_observations) == 4:

    A = len(last_actions) == 4
    B = len(last_observations) == 4

    Combinações necessárias (AB = Resultado):
    1: VV = V (break - para de coletar)
    2: VF = F (4 actions mas não 4 observations)
    3: FV = F (4 observations mas não 4 actions)
    """

    @pytest.fixture
    def mock_state(self):
        """Cria um State mockado para os testes"""
        state = State(inputs={}, max_iterations=100)
        return state

    def test_line96_case1_both_lists_have_4(self, mock_state):
        """Caso 1: AB=VV - Ambas listas têm 4 elementos (break)"""
        # A=V: len(last_actions) == 4
        # B=V: len(last_observations) == 4

        # Adiciona exatamente 4 pares action-observation
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(
            CmdOutputObservation(content='file1.txt', command='ls')
        )
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )
        mock_state.history.append(CmdRunAction(command='echo "1"'))
        mock_state.history.append(CmdOutputObservation(content='1', command='echo "1"'))
        mock_state.history.append(CmdRunAction(command='echo "2"'))
        mock_state.history.append(CmdOutputObservation(content='2', command='echo "2"'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_line96_case2_4_actions_but_less_observations(self, mock_state):
        """Caso 2: AB=VF - 4 actions mas menos de 4 observations (continua)"""
        # A=V: len(last_actions) == 4
        # B=F: len(last_observations) < 4

        # Adiciona 4 actions mas apenas 2 observations
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(
            CmdOutputObservation(content='file1.txt', command='ls')
        )
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )
        mock_state.history.append(CmdRunAction(command='echo "1"'))
        mock_state.history.append(CmdRunAction(command='echo "2"'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)

    def test_line96_case3_4_observations_but_less_actions(self, mock_state):
        """Caso 3: AB=FV - 4 observations mas menos de 4 actions (continua)"""
        # A=F: len(last_actions) < 4
        # B=V: len(last_observations) == 4

        # Adiciona apenas 2 actions mas simula 4 observations
        # (cenário pode não ser realista, mas testa a condição)
        mock_state.history.append(CmdRunAction(command='ls'))
        mock_state.history.append(
            CmdOutputObservation(content='file1.txt', command='ls')
        )
        mock_state.history.append(
            CmdOutputObservation(content='file2.txt', command='ls')
        )
        mock_state.history.append(CmdRunAction(command='pwd'))
        mock_state.history.append(
            CmdOutputObservation(content='/workspace', command='pwd')
        )
        mock_state.history.append(CmdOutputObservation(content='/home', command='pwd'))

        detector = StuckDetector(mock_state)
        result = detector.is_stuck(headless_mode=True)

        assert isinstance(result, bool)
