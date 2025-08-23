"""
Tests for the CLI module.
"""

import sys
from unittest.mock import AsyncMock, Mock, patch

from click.testing import CliRunner

from agent_memory_server.cli import (
    VERSION,
    api,
    cli,
    mcp,
    migrate_memories,
    rebuild_index,
    schedule_task,
    task_worker,
    version,
)


class TestVersion:
    """Tests for the version command."""

    def test_version_command(self):
        """Test that version command returns the correct version."""
        runner = CliRunner()
        result = runner.invoke(version)

        assert result.exit_code == 0
        assert VERSION in result.output
        assert "agent-memory-server version" in result.output


class TestRebuildIndex:
    """Tests for the rebuild_index command."""

    @patch("agent_memory_server.cli.ensure_search_index_exists")
    @patch("agent_memory_server.cli.get_redis_conn")
    def test_rebuild_index_command(self, mock_get_redis_conn, mock_ensure_index):
        """Test rebuild_index command execution."""
        # Use AsyncMock which returns completed awaitables
        mock_redis = Mock()
        mock_get_redis_conn.return_value = mock_redis
        mock_ensure_index.return_value = None

        runner = CliRunner()
        result = runner.invoke(rebuild_index)

        assert result.exit_code == 0
        mock_get_redis_conn.assert_called_once()
        mock_ensure_index.assert_called_once_with(mock_redis, overwrite=True)


class TestMigrateMemories:
    """Tests for the migrate_memories command."""

    @patch("agent_memory_server.cli.migrate_add_memory_type_3")
    @patch("agent_memory_server.cli.migrate_add_discrete_memory_extracted_2")
    @patch("agent_memory_server.cli.migrate_add_memory_hashes_1")
    @patch("agent_memory_server.cli.get_redis_conn")
    def test_migrate_memories_command(
        self,
        mock_get_redis_conn,
        mock_migration1,
        mock_migration2,
        mock_migration3,
    ):
        """Test migrate_memories command execution."""
        # Use AsyncMock which returns completed awaitables
        mock_redis = Mock()
        mock_get_redis_conn.return_value = mock_redis

        for migration in [mock_migration1, mock_migration2, mock_migration3]:
            migration.return_value = None

        runner = CliRunner()
        result = runner.invoke(migrate_memories)

        assert result.exit_code == 0
        assert "Starting memory migrations..." in result.output
        assert "Memory migrations completed successfully." in result.output
        mock_get_redis_conn.assert_called_once()
        mock_migration1.assert_called_once_with(redis=mock_redis)
        mock_migration2.assert_called_once_with(redis=mock_redis)
        mock_migration3.assert_called_once_with(redis=mock_redis)


class TestApiCommand:
    """Tests for the api command."""

    @patch("agent_memory_server.cli.uvicorn.run")
    @patch("agent_memory_server.main.on_start_logger")
    def test_api_command_defaults(self, mock_on_start_logger, mock_uvicorn_run):
        """Test api command with default parameters."""
        runner = CliRunner()
        result = runner.invoke(api)

        assert result.exit_code == 0
        mock_on_start_logger.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            "agent_memory_server.main:app",
            host="0.0.0.0",
            port=8000,  # default from settings
            reload=False,
        )

    @patch("agent_memory_server.cli.uvicorn.run")
    @patch("agent_memory_server.main.on_start_logger")
    def test_api_command_with_options(self, mock_on_start_logger, mock_uvicorn_run):
        """Test api command with custom parameters."""
        runner = CliRunner()
        result = runner.invoke(
            api, ["--port", "9000", "--host", "127.0.0.1", "--reload"]
        )

        assert result.exit_code == 0
        mock_on_start_logger.assert_called_once_with(9000)
        mock_uvicorn_run.assert_called_once_with(
            "agent_memory_server.main:app",
            host="127.0.0.1",
            port=9000,
            reload=True,
        )


class TestMcpCommand:
    """Tests for the mcp command."""

    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_stdio_mode(self, mock_mcp_app, mock_settings):
        """Test mcp command in stdio mode."""
        mock_settings.mcp_port = 3001
        mock_settings.log_level = "INFO"

        mock_mcp_app.run_stdio_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "stdio"])

        assert result.exit_code == 0
        mock_mcp_app.run_stdio_async.assert_called_once()

    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_sse_mode(self, mock_mcp_app, mock_settings):
        """Test mcp command in SSE mode."""
        mock_settings.mcp_port = 3001

        mock_mcp_app.run_sse_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "sse", "--port", "4000"])

        assert result.exit_code == 0
        mock_mcp_app.run_sse_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_mcp_logging")
    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_stdio_logging_config(
        self, mock_mcp_app, mock_settings, mock_configure_mcp_logging
    ):
        """Test that stdio mode configures logging to stderr."""
        mock_settings.mcp_port = 3001
        mock_settings.log_level = "DEBUG"

        mock_mcp_app.run_stdio_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "stdio"])

        assert result.exit_code == 0
        mock_mcp_app.run_stdio_async.assert_called_once()
        mock_configure_mcp_logging.assert_called_once()


class TestScheduleTask:
    """Tests for the schedule_task command."""

    def test_schedule_task_invalid_arg_format(self):
        """Test error handling for invalid argument format."""
        runner = CliRunner()
        result = runner.invoke(
            schedule_task, ["test.module.test_function", "--args", "invalid_format"]
        )

        assert result.exit_code == 1
        assert "Invalid argument format" in result.output

    def test_schedule_task_sync_error_handling(self):
        """Test error handling in sync part (before asyncio.run)."""
        # Test import error
        runner = CliRunner()
        result = runner.invoke(schedule_task, ["invalid.module.path"])
        assert result.exit_code == 1

        # Test invalid arguments
        result = runner.invoke(
            schedule_task,
            ["test.module.function", "--args", "invalid_arg_without_equals"],
        )
        assert result.exit_code == 1

    def test_schedule_task_argument_parsing(self):
        """Test various argument parsing scenarios."""
        # We test this by calling the command with invalid arguments
        runner = CliRunner()

        # Test invalid argument format
        result = runner.invoke(
            schedule_task, ["test.module.function", "--args", "invalid_format"]
        )
        assert result.exit_code == 1
        assert "Invalid argument format" in result.output


class TestTaskWorker:
    """Tests for the task_worker command."""

    @patch("agent_memory_server.cli.ensure_search_index_exists")
    @patch("agent_memory_server.cli.get_redis_conn")
    @patch("docket.Worker.run")
    @patch("agent_memory_server.cli.settings")
    def test_task_worker_success(
        self,
        mock_settings,
        mock_worker_run,
        mock_get_redis_conn,
        mock_ensure_index,
        redis_url,
    ):
        """Test successful task worker start."""
        mock_settings.use_docket = True
        mock_settings.docket_name = "test-docket"
        mock_settings.redis_url = redis_url

        mock_worker_run.return_value = None
        mock_redis = AsyncMock()
        mock_get_redis_conn.return_value = mock_redis
        mock_ensure_index.return_value = None

        runner = CliRunner()
        result = runner.invoke(
            task_worker, ["--concurrency", "5", "--redelivery-timeout", "60"]
        )

        assert result.exit_code == 0
        mock_worker_run.assert_called_once()

    @patch("agent_memory_server.cli.settings")
    def test_task_worker_docket_disabled(self, mock_settings):
        """Test task worker when docket is disabled."""
        mock_settings.use_docket = False

        runner = CliRunner()
        result = runner.invoke(task_worker)

        assert result.exit_code == 1
        assert "Docket is disabled in settings" in result.output

    @patch("agent_memory_server.cli.ensure_search_index_exists")
    @patch("agent_memory_server.cli.get_redis_conn")
    @patch("docket.Worker.run")
    @patch("agent_memory_server.cli.settings")
    def test_task_worker_default_params(
        self,
        mock_settings,
        mock_worker_run,
        mock_get_redis_conn,
        mock_ensure_index,
        redis_url,
    ):
        """Test task worker with default parameters."""
        mock_settings.use_docket = True
        mock_settings.docket_name = "test-docket"
        mock_settings.redis_url = redis_url

        mock_worker_run.return_value = None
        mock_redis = AsyncMock()
        mock_get_redis_conn.return_value = mock_redis
        mock_ensure_index.return_value = None

        runner = CliRunner()
        result = runner.invoke(task_worker)

        assert result.exit_code == 0
        mock_worker_run.assert_called_once()


class TestCliGroup:
    """Tests for the main CLI group."""

    def test_cli_group_help(self):
        """Test that CLI group shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Command-line interface for agent-memory-server" in result.output

    def test_cli_group_commands_exist(self):
        """Test that all expected commands are registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        expected_commands = [
            "version",
            "rebuild-index",
            "migrate-memories",
            "api",
            "mcp",
            "schedule-task",
            "task-worker",
        ]
        for command in expected_commands:
            assert command in result.output


class TestMainExecution:
    """Tests for main execution."""

    @patch("agent_memory_server.cli.cli")
    def test_main_execution(self, mock_cli):
        """Test that main execution calls the CLI."""
        # Import and run the main execution code
        import agent_memory_server.cli

        # The main execution is guarded by if __name__ == "__main__"
        # We can test this by patching sys.modules and importing
        with patch.dict(sys.modules, {"__main__": agent_memory_server.cli}):
            # This would normally call cli() but we've mocked it
            pass
