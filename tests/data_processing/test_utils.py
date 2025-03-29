# tests/data_processing/test_utils.py
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

# Adjust the import path based on how pytest discovers tests and the project structure
# Assuming tests are run from the project root directory
# If running pytest from root, src needs to be importable (e.g., via PYTHONPATH or project structure)
# Alternatively, adjust sys.path in a conftest.py or use relative imports if tests are structured differently.
# Assuming src is directly importable:
from src.data_processing import utils as data_utils

# --- Fixtures ---


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


# --- Tests for count_lines ---


def test_count_lines_wc_success(mocker):
    """Test count_lines when 'wc -l' succeeds."""
    mock_filepath = Path("dummy/file.txt")
    mocker.patch.object(mock_filepath, "is_file", return_value=True)
    # Mock subprocess.check_output to return typical wc output
    mocker.patch("subprocess.check_output", return_value="  123 file.txt\n")

    assert data_utils.count_lines(mock_filepath) == 123
    subprocess.check_output.assert_called_once_with(
        ["bash", "-c", f"wc -l < '{mock_filepath}'"], text=True, stderr=subprocess.PIPE
    )


def test_count_lines_wc_fail_fallback_python(mocker, temp_test_dir):
    """Test count_lines fallback to Python when 'wc -l' fails."""
    mock_filepath = temp_test_dir / "test_fallback.txt"
    # Create a dummy file with content
    file_content = "line 1\nline 2\nline 3\n"
    mock_filepath.write_text(file_content)

    # Mock subprocess.check_output to raise an error
    mocker.patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "wc")
    )

    # Mock print to check warning message (optional)
    mock_print = mocker.patch("builtins.print")

    assert data_utils.count_lines(mock_filepath) == 3
    subprocess.check_output.assert_called_once()  # Check it was called
    mock_print.assert_any_call(
        pytest.string_containing("Warning: 'wc -l' failed"),
    )


def test_count_lines_file_not_found():
    """Test count_lines raises FileNotFoundError for non-existent files."""
    non_existent_path = Path("non/existent/file.txt")
    with pytest.raises(FileNotFoundError):
        data_utils.count_lines(non_existent_path)


def test_count_lines_empty_file_python(temp_test_dir, mocker):
    """Test count_lines with an empty file using Python fallback."""
    mock_filepath = temp_test_dir / "empty.txt"
    mock_filepath.touch()  # Create empty file
    mocker.patch(
        "subprocess.check_output", side_effect=FileNotFoundError
    )  # Force fallback
    mocker.patch("builtins.print")  # Suppress warning print

    assert data_utils.count_lines(mock_filepath) == 0


# --- Tests for split_file ---


def test_split_file_basic(temp_test_dir, mocker):
    """Test basic file splitting."""
    input_dir = temp_test_dir / "input"
    output_dir = temp_test_dir / "output"
    input_dir.mkdir()
    input_filepath = input_dir / "source.txt"
    num_lines = 10
    num_chunks = 3
    lines_per_chunk = 4  # (10 + 3 - 1) // 3 = 4
    file_content = "".join([f"line {i+1}\n" for i in range(num_lines)])
    input_filepath.write_text(file_content)

    # Mock count_lines to avoid dependency on wc/python counting during split test
    mocker.patch("src.data_processing.utils.count_lines", return_value=num_lines)
    mock_print = mocker.patch("builtins.print")  # To check logging

    created_files = data_utils.split_file(
        input_filepath, output_dir, num_chunks, "source.txt"
    )

    assert len(created_files) == num_chunks
    assert output_dir.exists()

    # Check file names and content
    expected_files = [
        output_dir / "source.txt.000",
        output_dir / "source.txt.001",
        output_dir / "source.txt.002",
    ]
    assert set(created_files) == set(expected_files)

    # Check content of each chunk
    assert expected_files[0].read_text() == "line 1\nline 2\nline 3\nline 4\n"
    assert expected_files[1].read_text() == "line 5\nline 6\nline 7\nline 8\n"
    assert expected_files[2].read_text() == "line 9\nline 10\n"

    mock_print.assert_any_call(pytest.string_containing(f"Splitting {input_filepath}"))
    mock_print.assert_any_call(
        pytest.string_containing(f"Created {num_chunks} chunk files")
    )


def test_split_file_fewer_lines_than_chunks(temp_test_dir, mocker):
    """Test splitting when lines < chunks."""
    input_dir = temp_test_dir / "input"
    output_dir = temp_test_dir / "output"
    input_dir.mkdir()
    input_filepath = input_dir / "small.txt"
    num_lines = 2
    num_chunks = 5
    lines_per_chunk = 1  # (2 + 5 - 1) // 5 = 1
    file_content = "line 1\nline 2\n"
    input_filepath.write_text(file_content)

    mocker.patch("src.data_processing.utils.count_lines", return_value=num_lines)
    mocker.patch("builtins.print")

    created_files = data_utils.split_file(
        input_filepath, output_dir, num_chunks, "small.txt"
    )

    # Should create chunks until lines run out
    assert len(created_files) == 2
    assert (output_dir / "small.txt.000").read_text() == "line 1\n"
    assert (output_dir / "small.txt.001").read_text() == "line 2\n"


def test_split_file_empty_input(temp_test_dir, mocker):
    """Test splitting an empty file."""
    input_dir = temp_test_dir / "input"
    output_dir = temp_test_dir / "output"
    input_dir.mkdir()
    input_filepath = input_dir / "empty.txt"
    input_filepath.touch()

    mocker.patch("src.data_processing.utils.count_lines", return_value=0)
    mock_print = mocker.patch("builtins.print")

    created_files = data_utils.split_file(input_filepath, output_dir, 3, "empty.txt")

    assert created_files == []
    assert not any(output_dir.iterdir())  # Output dir should be empty
    mock_print.assert_any_call(
        pytest.string_containing("Input file") and pytest.string_containing("is empty")
    )


def test_split_file_invalid_chunks(temp_test_dir):
    """Test split_file raises error for invalid num_chunks."""
    input_filepath = temp_test_dir / "dummy.txt"
    input_filepath.touch()
    with pytest.raises(ValueError, match="Number of chunks must be at least 1"):
        data_utils.split_file(input_filepath, temp_test_dir, 0, "dummy.txt")
    with pytest.raises(ValueError, match="Number of chunks must be at least 1"):
        data_utils.split_file(input_filepath, temp_test_dir, -1, "dummy.txt")


# --- Tests for merge_files ---


def test_merge_files_basic(temp_test_dir):
    """Test basic file merging."""
    input_dir = temp_test_dir / "inputs"
    output_filepath = temp_test_dir / "merged.txt"
    input_dir.mkdir()

    file1 = input_dir / "f1.txt"
    file2 = input_dir / "f2.txt"
    file3 = input_dir / "f3.txt"

    file1.write_text("Content 1\n")
    file2.write_text("Content 2\n")
    file3.write_text("Content 3\n")

    input_files = [file1, file2, file3]
    data_utils.merge_files(input_files, output_filepath)

    assert output_filepath.exists()
    assert output_filepath.read_text() == "Content 1\nContent 2\nContent 3\n"
    # Check input files still exist
    assert file1.exists()
    assert file2.exists()
    assert file3.exists()


def test_merge_files_delete_inputs(temp_test_dir):
    """Test merging with delete_inputs=True."""
    input_dir = temp_test_dir / "inputs"
    output_filepath = temp_test_dir / "merged_del.txt"
    input_dir.mkdir()

    file1 = input_dir / "f1_del.txt"
    file2 = input_dir / "f2_del.txt"
    file1.write_text("Delete 1\n")
    file2.write_text("Delete 2\n")

    input_files = [file1, file2]
    data_utils.merge_files(input_files, output_filepath, delete_inputs=True)

    assert output_filepath.exists()
    assert output_filepath.read_text() == "Delete 1\nDelete 2\n"
    # Check input files are deleted
    assert not file1.exists()
    assert not file2.exists()


def test_merge_files_some_missing(temp_test_dir, mocker):
    """Test merging when some input files are missing."""
    input_dir = temp_test_dir / "inputs"
    output_filepath = temp_test_dir / "merged_missing.txt"
    input_dir.mkdir()

    file1 = input_dir / "f1_exists.txt"
    file_missing = input_dir / "f_missing.txt"
    file3 = input_dir / "f3_exists.txt"

    file1.write_text("Exists 1\n")
    file3.write_text("Exists 3\n")

    input_files = [file1, file_missing, file3]
    mock_print = mocker.patch("builtins.print")
    data_utils.merge_files(input_files, output_filepath)

    assert output_filepath.exists()
    assert output_filepath.read_text() == "Exists 1\nExists 3\n"
    mock_print.assert_any_call(pytest.string_containing("not found during merge"))


# --- Tests for split_train_eval ---


@pytest.mark.parametrize(
    "num_lines, eval_ratio, expected_eval_lines",
    [
        (100, 0.1, 10),  # 1 out of 10
        (100, 0.2, 20),  # 1 out of 5
        (9, 0.1, 0),  # sample rate 10, index 9 never reached
        (10, 0.1, 1),  # sample rate 10, index 9 reached once
        (19, 0.1, 1),  # sample rate 10, index 9, 19 -> index 9 reached once
        (20, 0.1, 2),  # sample rate 10, index 9, 19 reached
        (5, 0.3, 1),  # sample rate 3, index 2 reached
        (6, 0.3, 2),  # sample rate 3, index 2, 5 reached
    ],
)
def test_split_train_eval_ratios(
    temp_test_dir, mocker, num_lines, eval_ratio, expected_eval_lines
):
    """Test split_train_eval with various ratios and line counts."""
    processed_filepath = temp_test_dir / "processed.tsv"
    train_filepath = temp_test_dir / "train.tsv"
    eval_filepath = temp_test_dir / "eval.tsv"

    # Create dummy processed file
    content = "".join([f"line {i}\n" for i in range(num_lines)])
    processed_filepath.write_text(content)

    mocker.patch("builtins.print")  # Suppress print output

    data_utils.split_train_eval(
        processed_filepath, train_filepath, eval_filepath, eval_ratio
    )

    assert train_filepath.exists()
    assert eval_filepath.exists()

    # Verify line counts
    # Use Python counting for simplicity in test, assuming count_lines works
    with train_filepath.open("r") as f:
        train_line_count = sum(1 for _ in f)
    with eval_filepath.open("r") as f:
        eval_line_count = sum(1 for _ in f)

    assert eval_line_count == expected_eval_lines
    assert train_line_count == num_lines - expected_eval_lines


def test_split_train_eval_invalid_ratio(temp_test_dir):
    """Test split_train_eval raises error for invalid eval_ratio."""
    processed_filepath = temp_test_dir / "processed.tsv"
    processed_filepath.touch()
    train_filepath = temp_test_dir / "train.tsv"
    eval_filepath = temp_test_dir / "eval.tsv"

    with pytest.raises(ValueError, match="Evaluation ratio must be between 0 and 1"):
        data_utils.split_train_eval(
            processed_filepath, train_filepath, eval_filepath, 0.0
        )
    with pytest.raises(ValueError, match="Evaluation ratio must be between 0 and 1"):
        data_utils.split_train_eval(
            processed_filepath, train_filepath, eval_filepath, 1.0
        )
    with pytest.raises(ValueError, match="Evaluation ratio must be between 0 and 1"):
        data_utils.split_train_eval(
            processed_filepath, train_filepath, eval_filepath, -0.1
        )


def test_split_train_eval_file_not_found(temp_test_dir):
    """Test split_train_eval raises error if processed file not found."""
    processed_filepath = temp_test_dir / "non_existent.tsv"
    train_filepath = temp_test_dir / "train.tsv"
    eval_filepath = temp_test_dir / "eval.tsv"

    with pytest.raises(FileNotFoundError):
        data_utils.split_train_eval(
            processed_filepath, train_filepath, eval_filepath, 0.1
        )
