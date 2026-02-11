import atexit
import logging
import shutil
from pathlib import Path

from . import backend

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from omegaconf import OmegaConf, ListConfig
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg

logger = logging.getLogger("aide")


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def run():
    cfg = load_cfg()
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    # Seed pre-specified solutions before the main loop
    seed_files = getattr(cfg.agent.search, "seed_solution_files", None)
    if seed_files:
        if not isinstance(seed_files, (list, tuple, ListConfig)):
            seed_files = [seed_files]
        num_drafts = cfg.agent.search.num_drafts
        if len(seed_files) >= num_drafts:
            raise ValueError(
                f"Number of seed solution files ({len(seed_files)}) must be "
                f"less than num_drafts ({num_drafts})."
            )
        for sol_path in seed_files:
            sol_path = Path(sol_path).resolve()
            if not sol_path.exists():
                raise FileNotFoundError(f"Seed solution file not found: {sol_path}")
            logger.info(f"Seeding solution from: {sol_path}")
            print(f"Seeding solution from: {sol_path}")
            code = sol_path.read_text()
            agent.seed_solution(code, exec_callback)
            save_run(cfg, journal)
            global_step = len(journal)

    def generate_live():
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"), prog, status
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    with Live(
        generate_live(),
        refresh_per_second=16,
        screen=True,
    ) as live:
        while global_step < cfg.agent.steps:
            agent.step(exec_callback=exec_callback)
            save_run(cfg, journal)
            global_step = len(journal)
            live.update(generate_live())
    interpreter.cleanup_session()

    if cfg.generate_report:
        print("Generating final report from journal...")
        report = journal2report(journal, task_desc, cfg.report)
        print(report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()
