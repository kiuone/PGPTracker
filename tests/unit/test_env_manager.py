"""
Unit tests for env_manager module.

Run with: pytest tests/unit/test_env_manager.py -v
"""

import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Import functions to test
from pgptracker.utils.env_manager import (
    detect_available_cores,
    detect_available_memory,
    check_conda_available,
    check_environment_exists,
    validate_environment,
    run_command,
    get_system_resources,
    ENV_MAP
)

class TestSystemDetection:
    """Tests for system resource detection functions."""
    
    def test_detect_available_cores(self):
        """Test CPU core detection."""
        
        cores = detect_available_cores()
        assert isinstance(cores, int)
        assert cores > 0
        #pass
    
    def test_detect_available_memory(self):
        """Test memory detection."""
        
        memory = detect_available_memory()
        assert isinstance(memory, float)
        assert memory >= 0  # May be 0 on non-Linux systems
        #pass
    
    def test_get_system_resources(self):
        """Test getting all system resources."""
        
        resources = get_system_resources()
        assert 'cores' in resources
        assert 'memory_gb' in resources
        assert isinstance(resources['cores'], int)
        assert isinstance(resources['memory_gb'], float)
        #pass


class TestCondaDetection:
    """Tests for conda availability checks."""

    def setup_method(self, method):
        """Clear the lru_cache before each test."""
        check_environment_exists.cache_clear() 
    
    # def teardown_method(self, method):
    #     """(Optional) Clear cache afterwards as well."""
    #     check_environment_exists.cache_clear()
    
    @patch('subprocess.run')
    def test_check_conda_available_success(self, mock_run):
        """Test when conda is available."""
        mock_run.return_value = MagicMock(returncode=0)
        
        
        result = check_conda_available()
        assert result is True
        mock_run.assert_called_once()
        #pass
    
    @patch('subprocess.run')
    def test_check_conda_available_failure(self, mock_run):
        """Test when conda is not available."""
        mock_run.side_effect = FileNotFoundError()
        
        
        result = check_conda_available()
        assert result is False
        #pass
    
    @patch('subprocess.run')
    def test_check_environment_exists_true(self, mock_run):
        """Test when environment exists."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="qiime2-amplicon-2025.10\npicrust2\npgptracker\n"
        )
        result = check_environment_exists("qiime2-amplicon-2025.10")
        assert result is True
        #pass
    
    @patch('subprocess.run')
    def test_check_environment_exists_false(self, mock_run):
        """Test when environment doesn't exist."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="base\nsome_other_env\n"
        )
        result = check_environment_exists("qiime2-amplicon-2025.10")
        assert result is False
        #pass


class TestValidateEnvironment:
    """Tests for environment validation."""
    
    @patch('pgptracker.utils.env_manager.check_conda_available')
    def test_validate_no_conda(self, mock_conda):
        """Test when conda is not available."""
        mock_conda.return_value = False
        with pytest.raises(RuntimeError, match="Conda is not available"):
            validate_environment("qiime")
        #pass
    
    @patch('pgptracker.utils.env_manager.check_conda_available')
    @patch('pgptracker.utils.env_manager.check_environment_exists')
    def test_validate_environment_missing(self, mock_env_exists, mock_conda):
        """Test when environment doesn't exist."""
        mock_conda.return_value = True
        mock_env_exists.return_value = False
        
        with pytest.raises(RuntimeError, match="not found"):
            validate_environment("qiime")
        #pass
    
    @patch('pgptracker.utils.env_manager.check_conda_available')
    @patch('pgptracker.utils.env_manager.check_environment_exists')
    def test_validate_environment_success(self, mock_env_exists, mock_conda):
        """Test successful environment validation."""
        mock_conda.return_value = True
        mock_env_exists.return_value = True
        
        result = validate_environment("qiime")
        assert result == "qiime2-amplicon-2025.10"
        #pass
    
    def test_validate_unknown_tool(self):
        """Test with unknown tool name."""
        
        with pytest.raises(ValueError, match="Unknown tool"):
            validate_environment("unknown_tool")
        #pass


class TestRunCommand:
    """Tests for running commands in conda environments."""
    
    @patch('pgptracker.utils.env_manager.validate_environment')
    @patch('subprocess.run')
    def test_run_command_success(self, mock_run, mock_validate):
        """Test successful command execution."""
        mock_validate.return_value = "qiime2-amplicon-2025.10"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="success",
            stderr=""
        )
        
        result = run_command("qiime", ["qiime", "--version"])
        assert result.returncode == 0
        mock_run.assert_called_once()
        #pass
    
    @patch('pgptracker.utils.env_manager.validate_environment')
    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run, mock_validate):
        """Test command execution failure."""
        mock_validate.return_value = "qiime2-amplicon-2025.10"
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error occurred"
        )
        
        with pytest.raises(subprocess.CalledProcessError):
            run_command("qiime", ["qiime", "invalid", "command"], check=True)
        #pass
    
    @patch('pgptracker.utils.env_manager.validate_environment')
    @patch('subprocess.run')
    def test_run_command_no_check(self, mock_run, mock_validate):
        """Test command execution with check=False."""
        mock_validate.return_value = "Picrust2"
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error"
        )
        result = run_command("Picrust2", ["some", "command"], check=False)
        assert result.returncode == 1
        # No exception should be raised
        #pass

@pytest.mark.slow
class TestIntegrationSystemResources:
    """Integration tests that actually check system (marked as slow)."""
    
    def test_actual_core_detection(self):
        """Test actual CPU core detection on system."""
        
        cores = detect_available_cores()
        assert cores >= 1  # Assume at least 1 core
        #pass
    
    def test_actual_conda_check(self, pytestconfig):
        """Test actual conda availability (slow, system-dependent)."""
        if not pytestconfig.getoption("--run-slow"):
            pytest.skip("Only run with --run-slow flag")
        result = check_conda_available()
        assert isinstance(result, bool)