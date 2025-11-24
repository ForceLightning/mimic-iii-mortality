# Standard Library
import os
import re

# Third-Party
from jsonargparse import auto_cli
from rich.progress import Progress


def main(
    mimic_iv_directory: str = "/mnt/x/Downloads/Compressed/mortality_prediction_mimic_iv/",
    raw_report_directory: str = "/mnt/x/Downloads/Compressed/note/",
    split_dir_names: tuple[str, str] = (
        "train_with_raw_report",
        "test_with_raw_report",
    ),
):
    train_files: list[str] = []
    for dirpath, _dirname, filenames in os.walk(
        os.path.join(mimic_iv_directory, split_dir_names[0])
    ):
        train_files += filenames

    test_files: list[str] = []
    for dirpath, _dirname, filenames in os.walk(
        os.path.join(mimic_iv_directory, split_dir_names[1])
    ):
        test_files += filenames

    for dirpath, _dirname, filenames in os.walk(raw_report_directory):
        with Progress() as progress:
            task = progress.add_task("Processing raw files...", total=len(filenames))
            for filename in filenames:
                progress.console.print(filename)
                substr: str = ""
                output_dir_path: str = ""
                with open(os.path.join(dirpath, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    try:
                        pattern = re.compile(
                            r'(?:(?:Chief|___) Complaint:\n)(.*)(?:")',
                            re.MULTILINE | re.UNICODE | re.DOTALL,
                        )
                        matches = pattern.search(text)
                        assert (
                            matches is not None
                        ), f"regex matches for file {filename} is empty!"
                        substrs = matches.groups()
                        assert (
                            len(substrs) != 0
                        ), f"regex groups for file {filename} is empty!"
                        substr = substrs[0]
                    except AssertionError:
                        pattern = re.compile(
                            r'(?:text\n")(.*)(\n")',
                            re.MULTILINE | re.UNICODE | re.DOTALL,
                        )
                        matches = pattern.search(text)
                        assert (
                            matches is not None
                        ), f"regex matches for file {filename} is empty!"
                        substrs = matches.groups()
                        assert (
                            len(substrs) != 0
                        ), f"regex groups for file {filename} is empty!"
                        substr = substrs[0]

                # Ignore the validation case.
                if filename in train_files:
                    output_dir_path = os.path.join(
                        mimic_iv_directory, split_dir_names[0]
                    )
                elif filename in test_files:
                    output_dir_path = os.path.join(
                        mimic_iv_directory, split_dir_names[1]
                    )

                output_filename = os.path.splitext(filename)[0] + ".txt"

                with open(
                    os.path.join(output_dir_path, "Report", output_filename),
                    "w+",
                    encoding="utf-8",
                ) as f:
                    _ = f.write(substr)
                progress.advance(task)


if __name__ == "__main__":
    _ = auto_cli(main, as_positional=False)
