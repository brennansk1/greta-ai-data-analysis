"""
Comprehensive tests for CLI module using pytest.
"""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, call
from typer.testing import CliRunner
from greta_cli.cli.main import app, init, run, report
from greta_cli.config import GretaConfig, DataConfig, PreprocessingConfig


class TestInitCommand:
    """Test init command functionality."""

    @patch('greta_cli.cli.main.save_config')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.validate_config')
    def test_init_command_basic(self, mock_validate, mock_load, mock_save):
        """Test basic init command."""
        mock_validate.return_value = []
        mock_load.return_value = Mock()

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create a dummy data file
            Path('test.csv').write_text('A,B\n1,2\n3,4')

            result = runner.invoke(app, ['init', '--data-source', 'test.csv'])

            assert result.exit_code == 0
            assert 'Config saved' in result.output
            mock_save.assert_called_once()

    @patch('greta_cli.cli.main.save_config')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.validate_config')
    def test_init_command_interactive(self, mock_validate, mock_load, mock_save):
        """Test init command with interactive mode."""
        mock_validate.return_value = []
        mock_load.return_value = Mock()

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('test.csv').write_text('A,B\n1,2')

            # Mock user input
            result = runner.invoke(app, ['init', '--interactive'], input='test.csv\ncsv\ntarget\n')

            assert result.exit_code == 0
            mock_save.assert_called_once()

    def test_init_command_missing_data_source(self):
        """Test init command without data source."""
        runner = CliRunner()
        result = runner.invoke(app, ['init'])

        assert result.exit_code == 1
        assert 'Data source is required' in result.output

    @patch('greta_cli.cli.main.save_config')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.validate_config')
    def test_init_command_with_all_options(self, mock_validate, mock_load, mock_save):
        """Test init command with all options specified."""
        mock_validate.return_value = []
        mock_load.return_value = Mock()

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('data.xlsx').write_text('dummy')

            result = runner.invoke(app, [
                'init',
                '--data-source', 'data.xlsx',
                '--data-type', 'excel',
                '--target-column', 'outcome',
                '--output', 'my_config.yml'
            ])

            assert result.exit_code == 0
            # Check that save_config was called with correct config
            call_args = mock_save.call_args
            config = call_args[0][0]
            assert config.data.source == 'data.xlsx'
            assert config.data.type == 'excel'
            assert config.data.target_column == 'outcome'

    @patch('greta_cli.cli.main.validate_config')
    def test_init_command_validation_warnings(self, mock_validate):
        """Test init command when validation shows warnings."""
        mock_validate.return_value = ['Invalid data type']

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('test.csv').write_text('A,B\n1,2')

            result = runner.invoke(app, ['init', '--data-source', 'test.csv'])

            assert result.exit_code == 0
            assert 'Configuration warnings:' in result.output
            assert 'Invalid data type' in result.output


class TestRunCommand:
    """Test run command functionality."""

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.format_results')
    @patch('greta_cli.cli.main.save_output')
    def test_run_command_basic(self, mock_save_output, mock_format, mock_load, mock_run):
        """Test basic run command."""
        mock_config = Mock()
        mock_load.return_value = mock_config
        mock_run.return_value = {'results': 'test'}
        mock_format.return_value = 'formatted results'

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create config file
            Path('config.yml').write_text('data:\n  source: test.csv')

            result = runner.invoke(app, ['run', '--config', 'config.yml'])

            assert result.exit_code == 0
            assert 'Analysis completed successfully' in result.output
            mock_run.assert_called_once_with(mock_config, {})

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.format_results')
    @patch('greta_cli.cli.main.save_output')
    def test_run_command_with_output_file(self, mock_save_output, mock_format, mock_load, mock_run):
        """Test run command with output file specified."""
        mock_config = Mock()
        mock_load.return_value = mock_config
        mock_run.return_value = {'results': 'test'}
        mock_format.return_value = 'formatted results'

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yml').write_text('data:\n  source: test.csv')

            result = runner.invoke(app, [
                'run',
                '--config', 'config.yml',
                '--output', 'results.json'
            ])

            assert result.exit_code == 0
            mock_save_output.assert_called_once_with('formatted results', 'results.json')

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    def test_run_command_config_load_failure(self, mock_load, mock_run):
        """Test run command when config loading fails."""
        mock_load.side_effect = FileNotFoundError("Config not found")

        runner = CliRunner()
        result = runner.invoke(app, ['run', '--config', 'missing.yml'])

        assert result.exit_code == 1
        assert 'Error loading config' in result.output

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    def test_run_command_analysis_failure(self, mock_load, mock_run):
        """Test run command when analysis fails."""
        mock_config = Mock()
        mock_load.return_value = mock_config
        mock_run.side_effect = Exception("Analysis error")

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yml').write_text('data:\n  source: test.csv')

            result = runner.invoke(app, ['run', '--config', 'config.yml'])

            assert result.exit_code == 1
            assert 'Error during analysis' in result.output

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.format_results')
    def test_run_command_with_overrides(self, mock_format, mock_load, mock_run):
        """Test run command with parameter overrides."""
        mock_config = Mock()
        mock_load.return_value = mock_config
        mock_run.return_value = {'results': 'test'}
        mock_format.return_value = 'formatted results'

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yml').write_text('data:\n  source: test.csv')

            override_json = '{"preprocessing": {"missing_strategy": "median"}}'
            result = runner.invoke(app, [
                'run',
                '--config', 'config.yml',
                '--override', override_json
            ])

            assert result.exit_code == 0
            # Check that overrides were parsed and passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            overrides = call_args[0][1]  # Second argument
            assert overrides == {"preprocessing": {"missing_strategy": "median"}}

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.format_results')
    def test_run_command_invalid_override_json(self, mock_format, mock_load, mock_run):
        """Test run command with invalid JSON overrides."""
        mock_config = Mock()
        mock_load.return_value = mock_config

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yml').write_text('data:\n  source: test.csv')

            result = runner.invoke(app, [
                'run',
                '--config', 'config.yml',
                '--override', 'invalid json'
            ])

            assert result.exit_code == 1
            assert 'Error parsing overrides' in result.output


class TestReportCommand:
    """Test report command functionality."""

    @patch('greta_cli.cli.main.generate_report')
    @patch('greta_cli.cli.main.save_output')
    def test_report_command_basic(self, mock_save_output, mock_generate):
        """Test basic report command."""
        mock_generate.return_value = 'Generated report content'

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create results file
            Path('results.json').write_text('{"hypotheses": [], "metadata": {}}')

            result = runner.invoke(app, ['report', '--input-file', 'results.json'])

            assert result.exit_code == 0
            assert 'Generated report content' in result.output
            mock_generate.assert_called_once()

    @patch('greta_cli.cli.main.generate_report')
    @patch('greta_cli.cli.main.save_output')
    def test_report_command_with_output_file(self, mock_save_output, mock_generate):
        """Test report command with output file."""
        mock_generate.return_value = 'Report content'

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('results.json').write_text('{"hypotheses": []}')

            result = runner.invoke(app, [
                'report',
                '--input-file', 'results.json',
                '--output', 'report.txt'
            ])

            assert result.exit_code == 0
            mock_save_output.assert_called_once_with('Report content', 'report.txt')

    @patch('greta_cli.cli.main.generate_report')
    def test_report_command_yaml_input(self, mock_generate):
        """Test report command with YAML input."""
        mock_generate.return_value = 'Report'

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('results.yaml').write_text('hypotheses: []')

            result = runner.invoke(app, [
                'report',
                '--input-file', 'results.yaml'
            ])

            assert result.exit_code == 0

    def test_report_command_missing_input_file(self):
        """Test report command with missing input file."""
        runner = CliRunner()
        result = runner.invoke(app, ['report', '--input-file', 'missing.json'])

        assert result.exit_code == 1
        assert 'Error loading results' in result.output

    @patch('greta_cli.cli.main.generate_report')
    def test_report_command_invalid_json(self, mock_generate):
        """Test report command with invalid JSON."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('invalid.json').write_text('not valid json')

            result = runner.invoke(app, ['report', '--input-file', 'invalid.json'])

            assert result.exit_code == 1
            assert 'Error loading results' in result.output

    @patch('greta_cli.cli.main.generate_report')
    def test_report_command_different_formats(self, mock_generate):
        """Test report command with different output formats."""
        mock_generate.return_value = 'Report'

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('results.json').write_text('{"hypotheses": []}')

            # Test markdown format
            result = runner.invoke(app, [
                'report',
                '--input-file', 'results.json',
                '--format', 'markdown'
            ])

            assert result.exit_code == 0
            mock_generate.assert_called_with({'hypotheses': []}, 'markdown')

    @patch('greta_cli.cli.main.generate_report')
    def test_report_command_generation_failure(self, mock_generate):
        """Test report command when report generation fails."""
        mock_generate.side_effect = Exception("Generation error")

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('results.json').write_text('{"hypotheses": []}')

            result = runner.invoke(app, ['report', '--input-file', 'results.json'])

            assert result.exit_code == 1
            assert 'Error generating report' in result.output


# Integration tests
class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @patch('greta_cli.cli.main.run_analysis_pipeline')
    @patch('greta_cli.cli.main.load_config')
    @patch('greta_cli.cli.main.format_results')
    @patch('greta_cli.cli.main.save_config')
    @patch('greta_cli.cli.main.validate_config')
    @patch('greta_cli.cli.main.generate_report')
    def test_full_cli_workflow(self, mock_generate, mock_validate, mock_save, mock_format, mock_load, mock_run):
        """Test complete CLI workflow from init to report."""
        from greta_cli.config import GretaConfig, DataConfig

        mock_config = GretaConfig(data=DataConfig(source='data.csv', type='csv', target_column='E'))
        mock_load.return_value = mock_config
        mock_validate.return_value = []
        mock_run.return_value = {
            'hypotheses': [{'features': [0], 'significance': 0.9}],
            'metadata': {'data_shape': (100, 5)}
        }
        mock_format.return_value = '{"results": "test"}'
        mock_generate.return_value = 'Generated report'

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create data file
            Path('data.csv').write_text('A,B,C,D,E\n1,2,3,4,5\n6,7,8,9,10')

            # Init
            result_init = runner.invoke(app, [
                'init',
                '--data-source', 'data.csv',
                '--target-column', 'E'
            ])
            assert result_init.exit_code == 0

            # Run
            result_run = runner.invoke(app, [
                'run',
                '--config', 'config.yml',
                '--output', 'results.json'
            ])
            assert result_run.exit_code == 0

            # Report
            result_report = runner.invoke(app, [
                'report',
                '--input-file', 'results.json',
                '--format', 'text'
            ])
            assert result_report.exit_code == 0

    def test_cli_help_commands(self):
        """Test CLI help commands."""
        runner = CliRunner()

        # Main help
        result = runner.invoke(app, ['--help'])
        assert result.exit_code == 0
        assert 'GRETA CLI' in result.output

        # Init help
        result = runner.invoke(app, ['init', '--help'])
        assert result.exit_code == 0
        assert 'Initialize a new Greta project' in result.output

        # Run help
        result = runner.invoke(app, ['run', '--help'])
        assert result.exit_code == 0
        assert 'Execute the analysis pipeline' in result.output

        # Report help
        result = runner.invoke(app, ['report', '--help'])
        assert result.exit_code == 0
        assert 'Generate a human-readable report' in result.output

    @patch('greta_cli.cli.main.validate_config')
    def test_cli_config_validation_integration(self, mock_validate):
        """Test config validation in CLI context."""
        mock_validate.return_value = ['Warning: Small population size']

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('data.csv').write_text('A,B\n1,2\n3,4')

            result = runner.invoke(app, [
                'init',
                '--data-source', 'data.csv'
            ])

            # Should still succeed but show warnings
            assert result.exit_code == 0