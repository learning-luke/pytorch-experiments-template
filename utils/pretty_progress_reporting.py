from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class PrettyProgressReporter:
    def __init__(self, metric_trackers, set_size_list, max_epochs, start_epoch, test):

        train_metric_tracker, val_metric_tracker, test_metric_tracker = metric_trackers
        self.train_set_size, self.val_set_size, self.test_set_size = set_size_list

        self.epoch_progress = Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            train_metric_tracker.get_metric_text_column(),
        )

        train_progress_dict = {
            **dict(description="[green]Training", total=self.train_set_size),
            **train_metric_tracker.get_current_iteration_metric_text_column_fields(),
        }

        val_progress_dict = {
            **dict(description="[yellow]Validation", total=self.val_set_size),
            **val_metric_tracker.get_current_iteration_metric_text_column_fields(),
        }

        self.progress_tracker = {}

        self.progress_tracker["training"] = self.epoch_progress.add_task(
            **train_progress_dict
        )
        self.progress_tracker["validation"] = self.epoch_progress.add_task(
            **val_progress_dict
        )

        for task in self.epoch_progress.tasks:
            print(task)

        self.overall_progress = Progress()

        total_iters = (max_epochs - start_epoch) * (
            self.train_set_size + self.val_set_size
        ) + self.test_set_size * test

        self.overall_task = self.overall_progress.add_task(
            "Experiment Progress",
            completed=(start_epoch) * (self.train_set_size + self.val_set_size),
            total=total_iters,
        )

        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(
                self.overall_progress,
                title="Experiment Progress",
                border_style="green",
                padding=(2, 2),
            ),
        )
        self.progress_table.add_row(
            Panel.fit(
                self.epoch_progress, title="[b]Jobs", border_style="red", padding=(1, 2)
            )
        )
        self.progress_table.add_row(train_metric_tracker.per_epoch_table)
        self.progress_table.add_row(val_metric_tracker.per_epoch_table)

    def update_progress_iter(self, metric_tracker, reset):

        if (
            metric_tracker.tracker_name == "testing"
            and "testing" not in self.progress_tracker
        ):

            test_progress_dict = {
                **dict(description="[red]Testing", total=self.test_set_size),
                **metric_tracker.get_current_iteration_metric_text_column_fields(),
            }

            self.progress_tracker["testing"] = self.epoch_progress.add_task(
                **test_progress_dict
            )

        if reset:
            self.epoch_progress.reset(
                task_id=self.progress_tracker[metric_tracker.tracker_name]
            )

        iter_update_dict = {
            **dict(
                task_id=self.progress_tracker[metric_tracker.tracker_name], advance=1
            ),
            **metric_tracker.get_current_iteration_metric_text_column_fields(),
        }
        self.epoch_progress.update(**iter_update_dict)
        self.overall_progress.advance(self.overall_task, advance=1)
