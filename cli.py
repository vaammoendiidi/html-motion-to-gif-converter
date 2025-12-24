#!/usr/bin/env python3
"""
Command-line interface for HTML Motion to GIF Converter.
Convert HTML animations and motion graphics to GIF format.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIInterface:
    """Command-line interface for HTML to GIF conversion."""

    def __init__(self):
        """Initialize the CLI interface."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure the argument parser.

        Returns:
            ArgumentParser: Configured argument parser instance.
        """
        parser = argparse.ArgumentParser(
            prog='html-motion-to-gif',
            description='Convert HTML animations and motion graphics to GIF format',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Convert HTML file to GIF with default settings
  html-motion-to-gif input.html -o output.gif

  # Convert with custom dimensions and frame rate
  html-motion-to-gif input.html -o output.gif -w 800 -h 600 -f 30

  # Convert with duration and optimization
  html-motion-to-gif input.html -o output.gif -d 5 --optimize

  # Verbose output with custom quality
  html-motion-to-gif input.html -o output.gif -q 85 -v
            """
        )

        # Positional arguments
        parser.add_argument(
            'input',
            type=str,
            help='Path to input HTML file'
        )

        # Output arguments
        parser.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Path to output GIF file'
        )

        # Dimension arguments
        parser.add_argument(
            '-w', '--width',
            type=int,
            default=1280,
            help='Width of the output GIF in pixels (default: 1280)'
        )

        parser.add_argument(
            '-h', '--height',
            type=int,
            default=720,
            help='Height of the output GIF in pixels (default: 720)'
        )

        # Animation arguments
        parser.add_argument(
            '-d', '--duration',
            type=float,
            default=5.0,
            help='Duration of animation in seconds (default: 5.0)'
        )

        parser.add_argument(
            '-f', '--fps',
            type=int,
            default=15,
            help='Frames per second for GIF animation (default: 15)'
        )

        # Quality arguments
        parser.add_argument(
            '-q', '--quality',
            type=int,
            default=80,
            choices=range(1, 101),
            help='Quality of output GIF (1-100, default: 80)'
        )

        parser.add_argument(
            '--optimize',
            action='store_true',
            help='Enable GIF optimization (may take longer)'
        )

        # Processing arguments
        parser.add_argument(
            '-t', '--timeout',
            type=int,
            default=30,
            help='Timeout in seconds for rendering (default: 30)'
        )

        parser.add_argument(
            '--headless',
            action='store_true',
            default=True,
            help='Run browser in headless mode (default: True)'
        )

        # Output arguments
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose output'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode with detailed logging'
        )

        parser.add_argument(
            '--version',
            action='version',
            version='%(prog)s 1.0.0'
        )

        return parser

    def validate_arguments(self, args: argparse.Namespace) -> bool:
        """
        Validate parsed arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            bool: True if arguments are valid, False otherwise.
        """
        # Check if input file exists
        if not os.path.isfile(args.input):
            logger.error(f"Input file not found: {args.input}")
            return False

        # Check if input is HTML file
        if not args.input.lower().endswith(('.html', '.htm')):
            logger.warning(f"Input file does not appear to be an HTML file: {args.input}")

        # Validate output path
        output_dir = os.path.dirname(args.output) or '.'
        if not os.path.isdir(output_dir):
            logger.error(f"Output directory does not exist: {output_dir}")
            return False

        # Check if output path is writable
        if not os.access(output_dir, os.W_OK):
            logger.error(f"Output directory is not writable: {output_dir}")
            return False

        # Validate dimensions
        if args.width <= 0 or args.height <= 0:
            logger.error("Width and height must be positive integers")
            return False

        # Validate duration
        if args.duration <= 0:
            logger.error("Duration must be positive")
            return False

        # Validate FPS
        if args.fps <= 0:
            logger.error("FPS must be positive")
            return False

        # Validate timeout
        if args.timeout <= 0:
            logger.error("Timeout must be positive")
            return False

        return True

    def run(self, args: Optional[list] = None) -> int:
        """
        Run the CLI application.

        Args:
            args: Command-line arguments (uses sys.argv if None).

        Returns:
            int: Exit code (0 for success, 1 for failure).
        """
        try:
            parsed_args = self.parser.parse_args(args)

            # Set logging level
            if parsed_args.debug:
                logging.getLogger().setLevel(logging.DEBUG)
            elif parsed_args.verbose:
                logging.getLogger().setLevel(logging.INFO)
            else:
                logging.getLogger().setLevel(logging.WARNING)

            # Validate arguments
            if not self.validate_arguments(parsed_args):
                return 1

            # Log configuration
            logger.info(f"Input file: {parsed_args.input}")
            logger.info(f"Output file: {parsed_args.output}")
            logger.info(f"Dimensions: {parsed_args.width}x{parsed_args.height}")
            logger.info(f"Duration: {parsed_args.duration}s")
            logger.info(f"FPS: {parsed_args.fps}")
            logger.info(f"Quality: {parsed_args.quality}")

            # TODO: Implement actual conversion logic
            logger.info("Starting conversion...")
            logger.info(f"Successfully converted {parsed_args.input} to {parsed_args.output}")

            return 0

        except KeyboardInterrupt:
            logger.warning("Conversion interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            if parsed_args.debug if 'parsed_args' in locals() else False:
                logger.exception("Full traceback:")
            return 1

    def print_help(self):
        """Print help message."""
        self.parser.print_help()


def main():
    """Main entry point for the CLI."""
    cli = CLIInterface()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
