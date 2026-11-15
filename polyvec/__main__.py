"""Command line: ``python -m polyvec <command>`` (installed as the ``polyvec`` script).

Commands::

    polyvec vectorize drawing.png -m flow -o drawing.svg
    polyvec keypoints drawing.png -o drawing.pts
    polyvec build --docker | --local
    polyvec info
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, available_methods, get_vectorizer, get_vectorizer_class
from .build import TOOLS, build_docker_image, build_local
from .errors import PolyvecError, VectorizationError
from .keypoints import PyramidalKeypointDetector
from .runners import DockerRunner, LocalRunner


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
def _cmd_vectorize(args: argparse.Namespace) -> int:
    options = {"runner": args.runner, "ink_threshold": args.ink_threshold,
               "invert": args.invert, "timeout": args.timeout, "workdir": args.workdir}
    inputs = {}
    if args.method_class.name == "flow":
        options["flow_iterations"] = args.flow_iterations
        if args.keypoints:
            inputs["keypoints"] = args.keypoints
    elif args.keypoints:
        print("note: --keypoints is only used by the flow method", file=sys.stderr)

    vectorizer = args.method_class(**options)
    drawing = vectorizer.vectorize(args.image, **inputs)
    output = Path(args.output) if args.output else Path(args.image).with_suffix(".svg")
    drawing.save(output)
    summary = f"{output}: {len(drawing)} strokes, {drawing.num_points} points"
    if "elapsed" in drawing.metadata:
        summary += (f" ({drawing.metadata['elapsed']:.1f} s via "
                    f"{drawing.metadata.get('runner', '?')} runner)")
    print(summary)
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
    vec.add_argument("--runner", choices=["auto", "local", "docker"], default="auto",
                     help="where the compiled backend runs (default: auto)")
    vec.add_argument("--timeout", type=float, help="abort the backend after this many seconds")
    vec.add_argument("--workdir", help="keep intermediate files and the backend log here")
    vec.set_defaults(handler=_cmd_vectorize)

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
