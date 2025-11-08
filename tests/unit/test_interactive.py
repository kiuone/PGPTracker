"""
Tests for interactive mode.

These tests mock user input to verify the interactive prompts
work correctly for all subcommands.
runs with: pytest tests/unit/test_interactive.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from pgptracker.interactive import (
    run_interactive_mode,
    _ask_yes_no,
    _ask_int,
    _ask_float,
    _ask_choice,
    _display_and_ask_resources
)

# --- Helper Function Tests ---

def test_ask_yes_no_yes():
    """Test _ask_yes_no returns True for 'y'."""
    with patch('builtins.input', return_value='y'):
        assert _ask_yes_no("Test?") is True

def test_ask_yes_no_no():
    """Test _ask_yes_no returns False for 'n'."""
    with patch('builtins.input', return_value='n'):
        assert _ask_yes_no("Test?") is False

def test_ask_yes_no_default_true():
    """Test _ask_yes_no uses default True on empty input."""
    with patch('builtins.input', return_value=''):
        assert _ask_yes_no("Test?", default=True) is True

def test_ask_yes_no_default_false():
    """Test _ask_yes_no uses default False on empty input."""
    with patch('builtins.input', return_value=''):
        assert _ask_yes_no("Test?", default=False) is False

def test_ask_yes_no_retry_on_invalid():
    """Test _ask_yes_no retries on invalid input."""
    with patch('builtins.input', side_effect=['invalid', 'y']):
        assert _ask_yes_no("Test?") is True

def test_ask_int_valid():
    """Test _ask_int accepts valid integer."""
    with patch('builtins.input', return_value='8'):
        assert _ask_int("Threads?", default=4) == 8

def test_ask_int_default():
    """Test _ask_int uses default on empty input."""
    with patch('builtins.input', return_value=''):
        assert _ask_int("Threads?", default=4) == 4

def test_ask_int_retry_on_invalid():
    """Test _ask_int retries on invalid input."""
    with patch('builtins.input', side_effect=['invalid', '8']):
        assert _ask_int("Threads?", default=4) == 8

def test_ask_float_valid():
    """Test _ask_float accepts valid float."""
    with patch('builtins.input', return_value='1.9'):
        assert _ask_float("NSTI?", default=1.7) == 1.9

def test_ask_float_default():
    """Test _ask_float uses default on empty input."""
    with patch('builtins.input', return_value=''):
        assert _ask_float("NSTI?", default=1.7) == 1.7

def test_ask_float_retry_on_invalid():
    """Test _ask_float retries on invalid input."""
    with patch('builtins.input', side_effect=['invalid', '1.9']):
        assert _ask_float("NSTI?", default=1.7) == 1.9

def test_ask_choice_valid():
    """Test _ask_choice accepts valid selection."""
    choices = ['Lv1', 'Lv2', 'Lv3']
    with patch('builtins.input', return_value='2'):
        assert _ask_choice("Level?", choices, default='Lv3') == 'Lv2'

def test_ask_choice_default():
    """Test _ask_choice uses default on empty input."""
    choices = ['Lv1', 'Lv2', 'Lv3']
    with patch('builtins.input', return_value=''):
        assert _ask_choice("Level?", choices, default='Lv3') == 'Lv3'

def test_ask_choice_retry_on_invalid():
    """Test _ask_choice retries on invalid input."""
    choices = ['Lv1', 'Lv2', 'Lv3']
    with patch('builtins.input', side_effect=['99', '1']):
        assert _ask_choice("Level?", choices, default='Lv3') == 'Lv1'

@patch('pgptracker.interactive.detect_available_cores', return_value=8)
@patch('pgptracker.interactive.detect_available_memory', return_value=64)
def test_display_and_ask_resources(mock_mem, mock_cores):
    """Test _display_and_ask_resources displays info and asks for threads."""
    with patch('builtins.input', return_value='4'):
        result = _display_and_ask_resources()
        assert result == 4
        mock_cores.assert_called_once()
        mock_mem.assert_called_once()

# --- Interactive Mode Tests ---

def test_interactive_mode_quit():
    """Test quitting interactive mode."""
    with patch('builtins.input', return_value='q'):
        assert run_interactive_mode() == 0

def test_interactive_mode_invalid_then_quit():
    """Test invalid choice then quit."""
    with patch('builtins.input', side_effect=['invalid', 'q']):
        assert run_interactive_mode() == 0

def test_interactive_mode_keyboard_interrupt():
    """Test Ctrl+C cancellation."""
    with patch('builtins.input', side_effect=KeyboardInterrupt):
        assert run_interactive_mode() == 1

@patch('pgptracker.interactive.export_command')
def test_interactive_export_command(mock_export):
    """Test selecting export command."""
    mock_export.return_value = 0
    inputs = [
        '2',  # Select export
        '/path/to/rep_seqs.qza',  # rep_seqs
        '/path/to/feature_table.qza',  # feature_table
        ''  # default output
    ]
    
    with patch('builtins.input', side_effect=inputs):
        with patch('pathlib.Path.exists', return_value=True):
            result = run_interactive_mode()
            assert result == 0
            mock_export.assert_called_once()