"""Command line: ``python -m polyvec <command>`` (installed as the ``polyvec`` script).

Commands::

    polyvec vectorize drawing.png -m flow -o drawing.svg
    polyvec vectorize note.png -m text --base flow -d paddleocr --verify --attempts 4
    polyvec keypoints drawing.png -o drawing.pts
    polyvec words note.png -d paddleocr -o note.words.json
    polyvec build --docker | --local
    polyvec info
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, available_methods, get_vectorizer_class
from .drawing import VectorDrawing
from .build import TOOLS, build_docker_image, build_local
from .errors import PolyvecError, VectorizationError
from .keypoints import PyramidalKeypointDetector
from . import _native
from .runners import RUNNER_NAMES, DockerRunner, LocalRunner
from .text import TextVectorizer, available_detectors


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.handler(args)
    except VectorizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PolyvecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- commands
def _text_vectorizer(args: argparse.Namespace) -> "TextVectorizer":
    base = {"runner": args.runner, "ink_threshold": args.ink_threshold,
            "timeout": args.timeout, "workdir": args.workdir}
    if args.base == "flow":
        base["flow_iterations"] = args.flow_iterations
    return TextVectorizer(
        args.base, word_detector=args.detector, direction=args.direction, scale=args.scale,
        trace_page=not args.text_only, verify=args.verify, max_text_error=args.max_text_error,
        max_shape_error=args.max_shape_error, attempts=args.attempts, selection=args.select,
        invert=args.invert, **base)


def _cmd_vectorize(args: argparse.Namespace) -> int:
    options = {"runner": args.runner, "ink_threshold": args.ink_threshold,
               "invert": args.invert, "timeout": args.timeout, "workdir": args.workdir}
    inputs = {}
    if args.method_class.name == "text":
        vectorizer = _text_vectorizer(args)
        if args.words:
            inputs["words"] = args.words
        return _finish(args, vectorizer.vectorize(args.image, **inputs))
    if args.method_class.name == "flow":
        options["flow_iterations"] = args.flow_iterations
        if args.keypoints:
            inputs["keypoints"] = args.keypoints
    elif args.keypoints:
        print("note: --keypoints is only used by the flow method", file=sys.stderr)

    vectorizer = args.method_class(**options)
    return _finish(args, vectorizer.vectorize(args.image, **inputs))


def _finish(args: argparse.Namespace, drawing: "VectorDrawing") -> int:
    output = Path(args.output) if args.output else Path(args.image).with_suffix(".svg")
    drawing.save(output)
    summary = f"{output}: {len(drawing)} strokes, {drawing.num_points} points"
    if "elapsed" in drawing.metadata:
        summary += (f" ({drawing.metadata['elapsed']:.1f} s via "
                    f"{drawing.metadata.get('runner', '?')} runner)")
    if "words" in drawing.metadata:
        words = drawing.metadata["words"]
        summary += f"; {len(words)} words"
        if args.verify:
            summary += f", {sum(w['passed'] for w in words)} passed re-reading"
    print(summary)
    return 0


def _cmd_words(args: argparse.Namespace) -> int:
    vectorizer = TextVectorizer(args.base, word_detector=args.detector, direction=args.direction,
                                invert=args.invert, runner="auto")
    words = vectorizer.detect_words(args.image)
    output = Path(args.output) if args.output else Path(args.image).with_suffix(".words.json")
    words.save(output)
    print(f"{output}: {words!r}")
    return 0


def _cmd_keypoints(args: argparse.Namespace) -> int:
    detector = PyramidalKeypointDetector(device=args.device, threshold=args.threshold)
    keypoints = detector.detect(args.image)
    output = Path(args.output) if args.output else Path(args.image).with_suffix(".pts")
    keypoints.save(output)
    print(f"{output}: {keypoints!r}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if args.local:
        bin_dir = build_local(prefix=args.prefix, solver=args.solver, gurobi_home=args.gurobi_home,
                              jobs=args.jobs, cmake_args=args.cmake_arg)
        print(f"backends installed in {bin_dir}")
    else:
        if args.solver == "gurobi" or args.gurobi_home:
            print("error: the Docker image is built with HiGHS; Gurobi needs --local",
                  file=sys.stderr)
            return 1
        build_args = {"POLYVEC_BUILD_JOBS": str(args.jobs)} if args.jobs else {}
        image = build_docker_image(build_args=build_args, quiet=False)
        print(f"backend image ready: {image}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(f"polyvec {__version__}")
    print("methods:")
    for name in available_methods():
        cls = get_vectorizer_class(name)
        print(f"  {name:<6} {cls.title} - {cls.reference}")

    print("inprocess / subprocess runners (extension module from the wheel):")
    for tool in TOOLS:
        state = (_native.MODULES[tool] if _native.available(tool)
                 else f"not installed ({_native.import_error(tool)})")
        print(f"  {tool:<14} {state}")
    local, docker = LocalRunner(), DockerRunner()
    print("local runner:")
    for tool in TOOLS:
        print(f"  {tool:<14} {local.find(tool) or 'not built'}")
    print("docker runner:")
    if not docker.daemon_running():
        print("  Docker is not reachable")
    else:
        state = "built" if docker.image_exists() else "not built yet (built on first use)"
        print(f"  image {docker.image}: {state}")
    return 0


# --------------------------------------------------------------------------- parser
def _method(name: str) -> type:
    try:
        return get_vectorizer_class(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polyvec", description="Line-drawing vectorization "
                                     "with PolyVector fields (2019) and PolyVector flow (2021).")
    parser.add_argument("--version", action="version", version=f"polyvec {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    vec = commands.add_parser("vectorize", help="vectorize an image into an SVG")
    vec.add_argument("image", help="input image (PNG, JPEG, ...)")
    vec.add_argument("-m", "--method", dest="method_class", type=_method,
                     default=get_vectorizer_class("flow"),
                     help=f"one of: {', '.join(available_methods())} (default: flow)")
    vec.add_argument("-o", "--output", help="output SVG (default: image name with .svg)")
    vec.add_argument("-k", "--keypoints", help="flow: use this .pts file instead of the detector")
    vec.add_argument("--flow-iterations", type=int, default=1500,
                     help="flow: PolyVector-flow steps per curve (default: 1500)")
    vec.add_argument("--ink-threshold", type=int, metavar="1-254",
                     help="gray level below which a pixel is ink (upstream default: 165)")
    vec.add_argument("--invert", action="store_true", help="drawing is light on dark")
    vec.add_argument("--runner", choices=list(RUNNER_NAMES), default="auto",
                     help="where the compiled backend runs (default: auto)")
    vec.add_argument("--timeout", type=float, help="abort the backend after this many seconds")
    vec.add_argument("--workdir", help="keep intermediate files and the backend log here")
    text = vec.add_argument_group("text method (-m text)")
    text.add_argument("--base", choices=["flow", "field"], default="flow",
                      help="vectorizer for the page and each word (default: flow)")
    text.add_argument("-d", "--detector", default="doctr",
                      help=f"word detector: {', '.join(available_detectors())} (default: doctr)")
    text.add_argument("-w", "--words", help="use this word-box JSON instead of the detector")
    text.add_argument("--direction", default="ltr-ttb",
                      help="writing direction <primary>-<secondary>, axes ltr/rtl/ttb/btt "
                           "(default: ltr-ttb)")
    text.add_argument("--scale", type=float, default=4.0, help="upscaling of word crops (default: 4)")
    text.add_argument("--text-only", action="store_true", help="return only the word strokes")
    text.add_argument("--verify", action="store_true",
                      help="re-read each rasterized word with the OCR engine")
    text.add_argument("--max-text-error", type=float, default=0.25,
                      help="largest character error rate that passes (default: 0.25)")
    text.add_argument("--max-shape-error", type=float, help="largest shape error that passes")
    text.add_argument("--attempts", type=int, default=1, help="attempts per word (default: 1)")
    text.add_argument("--select", choices=["first", "best"], default="first",
                      help="first passing attempt, or best of all attempts (default: first)")
    vec.set_defaults(handler=_cmd_vectorize)

    wd = commands.add_parser("words", help="detect word boxes (text stage 1) into a JSON file")
    wd.add_argument("image")
    wd.add_argument("-o", "--output", help="output JSON (default: image name with .words.json)")
    wd.add_argument("-d", "--detector", default="doctr",
                    help=f"word detector: {', '.join(available_detectors())} (default: doctr)")
    wd.add_argument("--direction", default="ltr-ttb", help="writing direction (default: ltr-ttb)")
    wd.add_argument("--base", default="field", help=argparse.SUPPRESS)
    wd.add_argument("--invert", action="store_true", help="writing is light on dark")
    wd.set_defaults(handler=_cmd_words)

    kp = commands.add_parser("keypoints", help="detect keypoints (flow stage 1) into a .pts file")
    kp.add_argument("image")
    kp.add_argument("-o", "--output", help="output .pts (default: image name with .pts)")
    kp.add_argument("--device", help="torch device, e.g. cpu or cuda (default: auto)")
    kp.add_argument("--threshold", type=float, default=0.5, help="heat-map threshold (default: 0.5)")
    kp.set_defaults(handler=_cmd_keypoints)

    build = commands.add_parser("build", help="compile the C++ backends")
    where = build.add_mutually_exclusive_group()
    where.add_argument("--docker", action="store_true", help="build the Docker image (default)")
    where.add_argument("--local", action="store_true", help="build with the local toolchain")
    build.add_argument("--solver", choices=["auto", "highs", "gurobi"], default="auto",
                       help="--local: solver for the flow backend (auto: Gurobi if "
                            "GUROBI_HOME points at one, else HiGHS)")
    build.add_argument("--gurobi-home", help="--local: Gurobi installation directory")
    build.add_argument("--jobs", type=int, help="parallel compile jobs (default: 4)")
    build.add_argument("--prefix", help="--local: install prefix (default: per-user directory)")
    build.add_argument("--cmake-arg", action="append", default=[], metavar="ARG",
                       help="--local: extra CMake configure argument (repeatable)")
    build.set_defaults(handler=_cmd_build)

    info = commands.add_parser("info", help="show methods and which backends are available")
    info.set_defaults(handler=_cmd_info)
    return parser


if __name__ == "__main__":
    sys.exit(main())
