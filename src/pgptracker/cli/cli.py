"""
PGPTracker CLI - Main entry point.

Orchestrates Stage 1 (Processing) and Stage 2 (Analysis) commands.
Cleaned to match current wrapper implementations.
"""

import argparse
import sys
import importlib.resources
import subprocess
from datetime import datetime
from pathlib import Path
from pgptracker.stage1_processing import pipeline_st1
from pgptracker.cli import subcommands
from pgptracker.cli.interactive import run_interactive_mode
from pgptracker.utils.profiling_tools.profiler import MemoryProfiler
from pgptracker.cli.subcommands import parent_parser
from pgptracker.utils.profiling_tools.profile_config import use_preset, get_config
from pgptracker.utils.profiling_tools.profile_reporter import generate_tsv_report, print_pretty_table
from pgptracker.utils.env_manager import check_environment_exists, ENV_MAP

def setup_command(args: argparse.Namespace) -> int:
    print("PGPTracker - Environment Setup")
    try:
        env_files_path = importlib.resources.files("pgptracker") / "environments"
    except Exception as e:
        print(f"Critical Error: Could not find 'environments' folder. {e}")
        return 1

    env_map = {
        ENV_MAP["qiime"]: "qiime2-amplicon-2025.10.yml",
        ENV_MAP["Picrust2"]: "picrust2.yml",
    }

    for env_name, yml_filename in env_map.items():
        yml_path = env_files_path / yml_filename
        if not yml_path.is_file():
            print(f"Error: {yml_filename} not found.")
            continue

        if check_environment_exists(env_name) and not args.force:
            print(f"[INFO] Environment '{env_name}' exists. Skipping.")
            continue
            
        cmd = ["conda", "env", "create", "--name", env_name, "-f", str(yml_path)]
        if args.force and check_environment_exists(env_name):
             cmd = ["conda", "env", "update", "--name", env_name, "-f", str(yml_path), "--prune"]
             
        try:
            print(f"Setting up {env_name}...")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Setup failed for {env_name}: {e}")
            return 1
            
    return 0

def gui_command(args: argparse.Namespace) -> int:
    """Launch Stage 2 GUI."""
    try:
        from pgptracker.gui import run_app
        run_app(port=args.port)
        return 0
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        return 1

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pgptracker", description="PGPTracker CLI")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Process (Stage 1)
    process_parser = subparsers.add_parser("process", parents=[parent_parser], help="Run full Stage 1 pipeline")
    _add_process_arguments(process_parser)
    process_parser.set_defaults(func=process_command)

    # Setup
    setup_parser = subparsers.add_parser("setup", help="Setup environments")
    setup_parser.add_argument("-f", "--force", action="store_true")
    setup_parser.set_defaults(func=setup_command)

    # GUI
    gui_parser = subparsers.add_parser("gui", help="Launch Data Explorer")
    gui_parser.add_argument("--port", type=int, default=8050)
    gui_parser.set_defaults(func=gui_command)

    # Subcommands registration
    subcommands.register_export_command(subparsers)
    subcommands.register_place_seqs_command(subparsers)
    subcommands.register_hsp_command(subparsers)
    subcommands.register_metagenome_command(subparsers)
    subcommands.register_classify_command(subparsers)
    subcommands.register_merge_command(subparsers)
    subcommands.register_stratify_pgpt_command(subparsers)
    subcommands.register_unstratify_pgpt_command(subparsers)
    subcommands.register_clr_command(subparsers)
    subcommands.register_analysis_command(subparsers)

    return parser

def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    input_group = parser.add_argument_group("input files")
    input_group.add_argument("--rep-seqs", metavar="PATH", help="Input sequences (.qza/.fna)")
    input_group.add_argument("--feature-table", metavar="PATH", help="Input table (.qza/.biom)")
    input_group.add_argument("--classifier-qza", metavar="PATH", help="Custom classifier path")

    params_group = parser.add_argument_group("parameters")
    params_group.add_argument("--max-nsti", type=float, default=1.7)
    params_group.add_argument("--chunk-size", type=int, default=1000)
    params_group.add_argument("--stratified", action="store_true")
    params_group.add_argument("-l", "--tax-level", default="Genus", choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    params_group.add_argument("--pgpt-level", default="Lv3", choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'])

    output_group = parser.add_argument_group("output")
    output_group.add_argument("-o", "--output", metavar="PATH")

    res_group = parser.add_argument_group("resources")
    res_group.add_argument("-t", "--threads", type=int)

    parser.add_argument("-i", "--interactive", action="store_true")

def process_command(args: argparse.Namespace) -> int:
    if args.interactive:
        return run_interactive_mode()
    if not args.rep_seqs or not args.feature_table:
        print("ERROR: --rep-seqs and --feature-table required.", file=sys.stderr)
        return 1
    return pipeline_st1.run_pipeline(args)

def main() -> int:
    """Main entry point."""
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1
    if len(sys.argv) == 2 and sys.argv[1] == '-i':
        return run_interactive_mode()
    
    args = parser.parse_args()
    
    if getattr(args, 'profile', None):
        print(f"[INFO] Profiling enabled: {args.profile}")
        use_preset(args.profile)
        MemoryProfiler.enable()

    exit_code = args.func(args)

    if getattr(args, 'profile', None) and MemoryProfiler.is_enabled():
        MemoryProfiler.disable()
        if len(MemoryProfiler.get_profiles()) > 0:
            output_dir = Path(args.output) if hasattr(args, 'output') and args.output else Path("results")
            generate_tsv_report(output_dir / f"profile_report_{datetime.now():%Y%m%d_%H%M%S}.tsv")
            if get_config().show_pretty_table:
                print_pretty_table()

    return exit_code

if __name__ == "__main__":
    sys.exit(main())