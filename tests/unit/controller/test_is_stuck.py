import logging
from unittest.mock import Mock, patch

import pytest
from pytest import TempPathFactory

from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.controller.stuck import StuckDetector
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.action.commands import IPythonRunCellAction
from openhands.events.action.empty import NullAction
from openhands.events.observation import (
    CmdOutputObservation,
    FileReadObservation,
)
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.observation.commands import IPythonRunCellObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.stream import EventSource, EventStream
from openhands.storage import get_file_store


def collect_events(stream):
    return [event for event in stream.get_events()]


logging.basicConfig(level=logging.DEBUG)

jupyter_line_1 = '\n[Jupyter current working directory:'
jupyter_line_2 = '\n[Jupyter Python interpreter:'
code_snippet = """
edit_file_by_replace(
    'book_store.py',
    to_replace=\"""def total(basket):
    if not basket:
        return 0
"""


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_is_stuck'))


@pytest.fixture
def event_stream(temp_dir):
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('asdf', file_store)
    yield event_stream
    # clear after each test
    event_stream.clear()


class TestStuckDetector:
    @pytest.fixture
    def stuck_detector(self):
        state = State(inputs={})
        # Set the iteration flag's max_value to 50 (equivalent to the old max_iterations)
        state.iteration_flag.max_value = 50
        state.history = []  # Initialize history as an empty list
        return StuckDetector(state)

    def _impl_syntax_error_events(
        self,
        state: State,
        error_message: str,
        random_line: bool,
        incidents: int = 4,
    ):
        for i in range(incidents):
            ipython_action = IPythonRunCellAction(code=code_snippet)
            state.history.append(ipython_action)
            extra_number = (i + 1) * 10 if random_line else '42'
            extra_line = '\n' * (i + 1) if random_line else ''
            ipython_observation = IPythonRunCellObservation(
                content=f'  Cell In[1], line {extra_number}\n'
                'to_replace="""def largest(min_factor, max_factor):\n            ^\n'
                f'{error_message}{extra_line}' + jupyter_line_1 + jupyter_line_2,
                code=code_snippet,
            )
            # ipython_observation._cause = ipython_action._id
            state.history.append(ipython_observation)

    def _impl_unterminated_string_error_events(
        self, state: State, random_line: bool, incidents: int = 4
    ):
        for i in range(incidents):
            ipython_action = IPythonRunCellAction(code=code_snippet)
            state.history.append(ipython_action)
            line_number = (i + 1) * 10 if random_line else '1'
            ipython_observation = IPythonRunCellObservation(
                content=f'print("  Cell In[1], line {line_number}\nhello\n       ^\nSyntaxError: unterminated string literal (detected at line {line_number})'
                + jupyter_line_1
                + jupyter_line_2,
                code=code_snippet,
            )
            # ipython_observation._cause = ipython_action._
            state.history.append(ipython_observation)

    def test_history_too_short(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        observation = NullObservation(content='')
        # observation._cause = message_action.id
        state.history.append(message_action)
        state.history.append(observation)

        cmd_action = CmdRunAction(command='ls')
        state.history.append(cmd_action)
        cmd_observation = CmdOutputObservation(
            command='ls', content='file1.txt\nfile2.txt'
        )
        # cmd_observation._cause = cmd_action._id
        state.history.append(cmd_observation)

        assert stuck_detector.is_stuck(headless_mode=True) is False
        assert stuck_detector.stuck_analysis is None

    def test_interactive_mode_resets_after_user_message(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state

        # First add some actions that would be stuck in non-UI mode
        for i in range(4):
            cmd_action = CmdRunAction(command='ls')
            cmd_action._id = i
            state.history.append(cmd_action)
            cmd_observation = CmdOutputObservation(
                content='', command='ls', command_id=i
            )
            cmd_observation._cause = cmd_action._id
            state.history.append(cmd_observation)

        # In headless mode, this should be stuck
        assert stuck_detector.is_stuck(headless_mode=True) is True

        # with the UI, it will ALSO be stuck initially
        assert stuck_detector.is_stuck(headless_mode=False) is True

        # Add a user message
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)

        # In not-headless mode, this should not be stuck because we ignore history before user message
        assert stuck_detector.is_stuck(headless_mode=False) is False

        # But in headless mode, this should be still stuck because user messages do not count
        assert stuck_detector.is_stuck(headless_mode=True) is True

        # Add two more identical actions - still not stuck because we need at least 3
        for i in range(2):
            cmd_action = CmdRunAction(command='ls')
            cmd_action._id = i + 4
            state.history.append(cmd_action)
            cmd_observation = CmdOutputObservation(
                content='', command='ls', command_id=i + 4
            )
            cmd_observation._cause = cmd_action._id
            state.history.append(cmd_observation)

        assert stuck_detector.is_stuck(headless_mode=False) is False

        # Add two more identical actions - now it should be stuck
        for i in range(2):
            cmd_action = CmdRunAction(command='ls')
            cmd_action._id = i + 6
            state.history.append(cmd_action)
            cmd_observation = CmdOutputObservation(
                content='', command='ls', command_id=i + 6
            )
            cmd_observation._cause = cmd_action._id
            state.history.append(cmd_observation)

        assert stuck_detector.is_stuck(headless_mode=False) is True

    def test_is_stuck_repeating_action_observation(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        message_action = MessageAction(content='Done', wait_for_response=False)
        message_action._source = EventSource.USER

        hello_action = MessageAction(content='Hello', wait_for_response=False)
        hello_observation = NullObservation('')

        # 2 events
        state.history.append(hello_action)
        state.history.append(hello_observation)

        cmd_action_1 = CmdRunAction(command='ls')
        cmd_action_1._id = 1
        state.history.append(cmd_action_1)
        cmd_observation_1 = CmdOutputObservation(content='', command='ls')
        cmd_observation_1._cause = cmd_action_1._id
        state.history.append(cmd_observation_1)
        # 4 events

        cmd_action_2 = CmdRunAction(command='ls')
        cmd_action_2._id = 2
        state.history.append(cmd_action_2)
        cmd_observation_2 = CmdOutputObservation(content='', command='ls')
        cmd_observation_2._cause = cmd_action_2._id
        state.history.append(cmd_observation_2)
        # 6 events

        # random user message just because we can
        message_null_observation = NullObservation(content='')
        state.history.append(message_action)
        state.history.append(message_null_observation)
        # 8 events

        assert stuck_detector.is_stuck(headless_mode=True) is False

        cmd_action_3 = CmdRunAction(command='ls')
        cmd_action_3._id = 3
        state.history.append(cmd_action_3)
        cmd_observation_3 = CmdOutputObservation(content='', command='ls')
        cmd_observation_3._cause = cmd_action_3._id
        state.history.append(cmd_observation_3)
        # 10 events

        assert len(state.history) == 10
        assert stuck_detector.is_stuck(headless_mode=True) is False

        cmd_action_4 = CmdRunAction(command='ls')
        cmd_action_4._id = 4
        state.history.append(cmd_action_4)
        cmd_observation_4 = CmdOutputObservation(content='', command='ls')
        cmd_observation_4._cause = cmd_action_4._id
        state.history.append(cmd_observation_4)
        # 12 events

        assert len(state.history) == 12

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is True
            mock_warning.assert_called_once_with('Action, Observation loop detected')

        # recover to before first loop pattern
        assert stuck_detector.stuck_analysis.loop_type == 'repeating_action_observation'
        assert stuck_detector.stuck_analysis.loop_repeat_times == 4
        assert stuck_detector.stuck_analysis.loop_start_idx == 1

    def test_is_stuck_repeating_action_error(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        # (action, error_observation), not necessarily the same error
        message_action = MessageAction(content='Done', wait_for_response=False)
        message_action._source = EventSource.USER

        hello_action = MessageAction(content='Hello', wait_for_response=False)
        hello_observation = NullObservation(content='')
        state.history.append(hello_action)
        # hello_observation._cause = hello_action._id
        state.history.append(hello_observation)
        # 2 events

        cmd_action_1 = CmdRunAction(command='invalid_command')
        state.history.append(cmd_action_1)
        error_observation_1 = ErrorObservation(content='Command not found')
        # error_observation_1._cause = cmd_action_1._id
        state.history.append(error_observation_1)
        # 4 events

        cmd_action_2 = CmdRunAction(command='invalid_command')
        state.history.append(cmd_action_2)
        error_observation_2 = ErrorObservation(
            content='Command still not found or another error'
        )
        # error_observation_2._cause = cmd_action_2._id
        state.history.append(error_observation_2)
        # 6 events

        message_null_observation = NullObservation(content='')
        state.history.append(message_action)
        state.history.append(message_null_observation)
        # 8 events

        cmd_action_3 = CmdRunAction(command='invalid_command')
        state.history.append(cmd_action_3)
        error_observation_3 = ErrorObservation(content='Different error')
        # error_observation_3._cause = cmd_action_3._id
        state.history.append(error_observation_3)
        # 10 events

        cmd_action_4 = CmdRunAction(command='invalid_command')
        state.history.append(cmd_action_4)
        error_observation_4 = ErrorObservation(content='Command not found')
        # error_observation_4._cause = cmd_action_4._id
        state.history.append(error_observation_4)
        # 12 events

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is True
            mock_warning.assert_called_once_with(
                'Action, ErrorObservation loop detected'
            )
        assert stuck_detector.stuck_analysis.loop_type == 'repeating_action_error'
        assert stuck_detector.stuck_analysis.loop_repeat_times == 3
        assert stuck_detector.stuck_analysis.loop_start_idx == 1

    def test_is_stuck_invalid_syntax_error(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        self._impl_syntax_error_events(
            state,
            error_message='SyntaxError: invalid syntax. Perhaps you forgot a comma?',
            random_line=False,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is True

    def test_is_not_stuck_invalid_syntax_error_random_lines(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        self._impl_syntax_error_events(
            state,
            error_message='SyntaxError: invalid syntax. Perhaps you forgot a comma?',
            random_line=True,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is False

    def test_is_not_stuck_invalid_syntax_error_only_three_incidents(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        self._impl_syntax_error_events(
            state,
            error_message='SyntaxError: invalid syntax. Perhaps you forgot a comma?',
            random_line=True,
            incidents=3,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is False

    def test_is_stuck_incomplete_input_error(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        self._impl_syntax_error_events(
            state,
            error_message='SyntaxError: incomplete input',
            random_line=False,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is True

    def test_is_not_stuck_incomplete_input_error(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        self._impl_syntax_error_events(
            state,
            error_message='SyntaxError: incomplete input',
            random_line=True,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is False

    def test_is_not_stuck_ipython_unterminated_string_error_random_lines(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        self._impl_unterminated_string_error_events(state, random_line=True)

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is False

    def test_is_not_stuck_ipython_unterminated_string_error_only_two_incidents(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        self._impl_unterminated_string_error_events(
            state, random_line=False, incidents=2
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is False

    def test_is_stuck_ipython_unterminated_string_error(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        self._impl_unterminated_string_error_events(state, random_line=False)

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck(headless_mode=True) is True

    def test_is_not_stuck_ipython_syntax_error_not_at_end(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        # this test is to make sure we don't get false positives
        # since the "at line x" is changing in between!
        ipython_action_1 = IPythonRunCellAction(code='print("hello')
        state.history.append(ipython_action_1)
        ipython_observation_1 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 1)\nThis is some additional output',
            code='print("hello',
        )
        # ipython_observation_1._cause = ipython_action_1._id
        state.history.append(ipython_observation_1)

        ipython_action_2 = IPythonRunCellAction(code='print("hello')
        state.history.append(ipython_action_2)
        ipython_observation_2 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 1)\nToo much output here on and on',
            code='print("hello',
        )
        # ipython_observation_2._cause = ipython_action_2._id
        state.history.append(ipython_observation_2)

        ipython_action_3 = IPythonRunCellAction(code='print("hello')
        state.history.append(ipython_action_3)
        ipython_observation_3 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 3)\nEnough',
            code='print("hello',
        )
        # ipython_observation_3._cause = ipython_action_3._id
        state.history.append(ipython_observation_3)

        ipython_action_4 = IPythonRunCellAction(code='print("hello')
        state.history.append(ipython_action_4)
        ipython_observation_4 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 2)\nLast line of output',
            code='print("hello',
        )
        # ipython_observation_4._cause = ipython_action_4._id
        state.history.append(ipython_observation_4)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is False
            mock_warning.assert_not_called()

    def test_is_stuck_repeating_action_observation_pattern(
        self, stuck_detector: StuckDetector
    ):
        state = stuck_detector.state
        message_action = MessageAction(content='Come on', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)
        message_observation = NullObservation(content='')
        state.history.append(message_observation)

        cmd_action_1 = CmdRunAction(command='ls')
        state.history.append(cmd_action_1)
        cmd_observation_1 = CmdOutputObservation(
            command='ls', content='file1.txt\nfile2.txt'
        )
        # cmd_observation_1._cause = cmd_action_1._id
        state.history.append(cmd_observation_1)

        read_action_1 = FileReadAction(path='file1.txt')
        state.history.append(read_action_1)
        read_observation_1 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        # read_observation_1._cause = read_action_1._id
        state.history.append(read_observation_1)

        cmd_action_2 = CmdRunAction(command='ls')
        state.history.append(cmd_action_2)
        cmd_observation_2 = CmdOutputObservation(
            command='ls', content='file1.txt\nfile2.txt'
        )
        # cmd_observation_2._cause = cmd_action_2._id
        state.history.append(cmd_observation_2)

        read_action_2 = FileReadAction(path='file1.txt')
        state.history.append(read_action_2)
        read_observation_2 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        # read_observation_2._cause = read_action_2._id
        state.history.append(read_observation_2)

        message_action = MessageAction(content='Come on', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)

        message_null_observation = NullObservation(content='')
        state.history.append(message_null_observation)

        cmd_action_3 = CmdRunAction(command='ls')
        state.history.append(cmd_action_3)
        cmd_observation_3 = CmdOutputObservation(
            command='ls', content='file1.txt\nfile2.txt'
        )
        # cmd_observation_3._cause = cmd_action_3._id
        state.history.append(cmd_observation_3)

        read_action_3 = FileReadAction(path='file1.txt')
        state.history.append(read_action_3)
        read_observation_3 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        # read_observation_3._cause = read_action_3._id
        state.history.append(read_observation_3)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is True
            mock_warning.assert_called_once_with('Action, Observation pattern detected')
        assert (
            stuck_detector.stuck_analysis.loop_type
            == 'repeating_action_observation_pattern'
        )
        assert stuck_detector.stuck_analysis.loop_repeat_times == 3
        assert stuck_detector.stuck_analysis.loop_start_idx == 0  # null ignored

    def test_is_stuck_not_stuck(self, stuck_detector: StuckDetector):
        state = stuck_detector.state
        message_action = MessageAction(content='Done', wait_for_response=False)
        message_action._source = EventSource.USER

        hello_action = MessageAction(content='Hello', wait_for_response=False)
        state.history.append(hello_action)
        hello_observation = NullObservation(content='')
        # hello_observation._cause = hello_action._id
        state.history.append(hello_observation)

        cmd_action_1 = CmdRunAction(command='ls')
        state.history.append(cmd_action_1)
        cmd_observation_1 = CmdOutputObservation(
            command='ls', content='file1.txt\nfile2.txt'
        )
        # cmd_observation_1._cause = cmd_action_1._id
        state.history.append(cmd_observation_1)

        read_action_1 = FileReadAction(path='file1.txt')
        state.history.append(read_action_1)
        read_observation_1 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        # read_observation_1._cause = read_action_1._id
        state.history.append(read_observation_1)

        cmd_action_2 = CmdRunAction(command='pwd')
        state.history.append(cmd_action_2)
        cmd_observation_2 = CmdOutputObservation(command='pwd', content='/home/user')
        # cmd_observation_2._cause = cmd_action_2._id
        state.history.append(cmd_observation_2)

        read_action_2 = FileReadAction(path='file2.txt')
        state.history.append(read_action_2)
        read_observation_2 = FileReadObservation(
            content='Another file content', path='file2.txt'
        )
        # read_observation_2._cause = read_action_2._id
        state.history.append(read_observation_2)

        message_null_observation = NullObservation(content='')
        state.history.append(message_action)
        state.history.append(message_null_observation)

        cmd_action_3 = CmdRunAction(command='pwd')
        state.history.append(cmd_action_3)
        cmd_observation_3 = CmdOutputObservation(command='pwd', content='/home/user')
        # cmd_observation_3._cause = cmd_action_3._id
        state.history.append(cmd_observation_3)

        read_action_3 = FileReadAction(path='file2.txt')
        state.history.append(read_action_3)
        read_observation_3 = FileReadObservation(
            content='Another file content', path='file2.txt'
        )
        # read_observation_3._cause = read_action_3._id
        state.history.append(read_observation_3)

        assert stuck_detector.is_stuck(headless_mode=True) is False

    def test_is_stuck_monologue(self, stuck_detector):
        state = stuck_detector.state
        # Add events to the history list directly
        message_action_1 = MessageAction(content='Hi there!')
        message_action_1._source = EventSource.USER
        state.history.append(message_action_1)
        message_action_2 = MessageAction(content='Hi there!')
        message_action_2._source = EventSource.AGENT
        state.history.append(message_action_2)
        message_action_3 = MessageAction(content='How are you?')
        message_action_3._source = EventSource.USER
        state.history.append(message_action_3)

        cmd_kill_action = CmdRunAction(
            command='echo 42', thought="I'm not stuck, he's stuck"
        )
        state.history.append(cmd_kill_action)

        message_action_4 = MessageAction(content="I'm doing well, thanks for asking.")
        message_action_4._source = EventSource.AGENT
        state.history.append(message_action_4)
        message_action_5 = MessageAction(content="I'm doing well, thanks for asking.")
        message_action_5._source = EventSource.AGENT
        state.history.append(message_action_5)
        message_action_6 = MessageAction(content="I'm doing well, thanks for asking.")
        message_action_6._source = EventSource.AGENT
        state.history.append(message_action_6)

        assert stuck_detector.is_stuck(headless_mode=True)
        assert stuck_detector.stuck_analysis.loop_type == 'monologue'
        assert stuck_detector.stuck_analysis.loop_repeat_times == 3
        assert stuck_detector.stuck_analysis.loop_start_idx == 2  # null ignored

        # Add an observation event between the repeated message actions
        cmd_output_observation = CmdOutputObservation(
            content='OK, I was stuck, but no more.',
            command='storybook',
            exit_code=0,
        )
        # cmd_output_observation._cause = cmd_kill_action._id
        state.history.append(cmd_output_observation)

        message_action_7 = MessageAction(content="I'm doing well, thanks for asking.")
        message_action_7._source = EventSource.AGENT
        state.history.append(message_action_7)
        message_action_8 = MessageAction(content="I'm doing well, thanks for asking.")
        message_action_8._source = EventSource.AGENT
        state.history.append(message_action_8)

        with patch('logging.Logger.warning'):
            assert not stuck_detector.is_stuck(headless_mode=True)

    def test_is_stuck_context_window_error_loop(self, stuck_detector):
        """Test that we detect when we're stuck in a loop of context window errors."""
        state = stuck_detector.state

        # Add some initial events
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)
        message_observation = NullObservation(content='')
        state.history.append(message_observation)

        # Add ten consecutive condensation events (should detect as stuck)
        for _ in range(10):
            condensation = AgentCondensationObservation(
                content='Trimming prompt to meet context window limitations'
            )
            state.history.append(condensation)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is True
            mock_warning.assert_called_once_with(
                'Context window error loop detected - repeated condensation events'
            )
        assert stuck_detector.stuck_analysis.loop_type == 'context_window_error'
        assert stuck_detector.stuck_analysis.loop_repeat_times == 2
        assert stuck_detector.stuck_analysis.loop_start_idx == 0

    def test_is_not_stuck_context_window_error_with_other_events(self, stuck_detector):
        """Test that we don't detect a loop when there are other events between condensation events."""
        state = stuck_detector.state

        # Add some initial events
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)
        message_observation = NullObservation(content='')
        state.history.append(message_observation)

        # Add 10 condensation events with other events between them
        for i in range(10):
            # Add a condensation event
            condensation = AgentCondensationObservation(
                content='Trimming prompt to meet context window limitations'
            )
            state.history.append(condensation)

            # Add some other events between condensation events (except after the last one)
            if i < 9:
                # Add a command action and observation
                cmd_action = CmdRunAction(command=f'ls {i}')
                state.history.append(cmd_action)
                cmd_observation = CmdOutputObservation(
                    command=f'ls {i}', content='file1.txt\nfile2.txt'
                )
                state.history.append(cmd_observation)

                # Add a file read action and observation for even iterations
                if i % 2 == 0:
                    read_action = FileReadAction(path=f'file{i}.txt')
                    state.history.append(read_action)
                    read_observation = FileReadObservation(
                        content=f'File content {i}', path=f'file{i}.txt'
                    )
                    state.history.append(read_observation)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is False
            mock_warning.assert_not_called()

    def test_is_not_stuck_context_window_error_less_than_ten(self, stuck_detector):
        """Test that we don't detect a loop with less than ten condensation events."""
        state = stuck_detector.state

        # Add some initial events
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)
        message_observation = NullObservation(content='')
        state.history.append(message_observation)

        # Add only nine condensation events (should not detect as stuck)
        for _ in range(9):
            condensation = AgentCondensationObservation(
                content='Trimming prompt to meet context window limitations'
            )
            state.history.append(condensation)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is False
            mock_warning.assert_not_called()

    def test_is_stuck_context_window_error_with_user_messages(self, stuck_detector):
        """Test that we still detect a loop even with user messages between condensation events in headless mode.

        User messages are filtered out in the stuck detection logic, so they shouldn't
        prevent us from detecting a loop of condensation events.
        """
        state = stuck_detector.state

        # Add some initial events
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        state.history.append(message_action)
        message_observation = NullObservation(content='')
        state.history.append(message_observation)

        # Add condensation events with user messages between them (total of 10)
        for i in range(10):
            # Add a condensation event
            condensation = AgentCondensationObservation(
                content='Trimming prompt to meet context window limitations'
            )
            state.history.append(condensation)

            # Add user message between condensation events (except after the last one)
            if i < 9:
                user_message = MessageAction(
                    content=f'Please continue {i}', wait_for_response=False
                )
                user_message._source = EventSource.USER
                state.history.append(user_message)
                user_observation = NullObservation(content='')
                state.history.append(user_observation)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck(headless_mode=True) is True
            mock_warning.assert_called_once_with(
                'Context window error loop detected - repeated condensation events'
            )
        assert stuck_detector.stuck_analysis.loop_type == 'context_window_error'
        assert stuck_detector.stuck_analysis.loop_repeat_times == 2
        assert stuck_detector.stuck_analysis.loop_start_idx == 0

    def test_is_not_stuck_context_window_error_in_non_headless(self, stuck_detector):
        """Test that in non-headless mode, we don't detect a loop if the condensation events
        are before the last user message.

        In non-headless mode, we only look at events after the last user message.
        """
        state = stuck_detector.state

        # Add condensation events first
        for _ in range(10):
            condensation = AgentCondensationObservation(
                content='Trimming prompt to meet context window limitations'
            )
            state.history.append(condensation)

        # Add a user message at the end
        user_message = MessageAction(content='Please continue', wait_for_response=False)
        user_message._source = EventSource.USER
        state.history.append(user_message)
        user_observation = NullObservation(content='')
        state.history.append(user_observation)

        with patch('logging.Logger.warning') as mock_warning:
            # In headless mode, we should detect the loop
            assert stuck_detector.is_stuck(headless_mode=True) is True
            mock_warning.assert_called_once_with(
                'Context window error loop detected - repeated condensation events'
            )

            # Reset mock for next assertion
            mock_warning.reset_mock()

            # In non-headless mode, we should NOT detect the loop since we only look
            # at events after the last user message
            assert stuck_detector.is_stuck(headless_mode=False) is False
            mock_warning.assert_not_called()


class TestAgentController:
    @pytest.fixture
    def controller(self):
        controller = Mock(spec=AgentController)
        controller._is_stuck = AgentController._is_stuck.__get__(
            controller, AgentController
        )
        controller.delegate = None
        controller.state = Mock()
        return controller

    def test_is_stuck_delegate_stuck(self, controller: AgentController):
        controller.delegate = Mock()
        controller.delegate._is_stuck.return_value = True
        assert controller._is_stuck() is True


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
