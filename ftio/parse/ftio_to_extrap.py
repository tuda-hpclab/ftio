"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import sys

try:
    from extrap.entities.callpath import Callpath
    from extrap.entities.coordinate import Coordinate
    from extrap.entities.experiment import Experiment
    from extrap.entities.metric import Metric
    from extrap.entities.parameter import Parameter
    from extrap.fileio import io_helper
    from extrap.modelers.model_generator import ModelGenerator
except ImportError as e:
    raise ImportError(
        "Extra-P is not installed. Install it with: pip install .[extrap-libs]"
    ) from e

from ftio.parse.print import Print


def build_experiment(records) -> Experiment:
    """Builds an Extra-P Experiment from FTIO records
    Same as extrap/fileio/file_reader/jsonlines_file_reader.py
    """
    experiment = Experiment()
    complete_data = {}

    # group values by (callpath, metric) and coordinate..
    for r in records:
        # ranks is -1 when the process count can't be read from the file name
        if r.ranks > 0:
            key = (Callpath(r.callpath), Metric(r.metric))
            coordinate = Coordinate(r.ranks)
            io_helper.append_to_repetition_dict(complete_data, key, coordinate, r.value)

    # one Measurement per (key, coordinate)
    io_helper.repetition_dict_to_experiment(complete_data, experiment)

    # param is static. analog as ioparse
    experiment.add_parameter(Parameter("Processes"))

    experiment.call_tree = io_helper.create_call_tree(experiment.callpaths)
    io_helper.validate_experiment(experiment)

    return experiment


def fit_models(experiment: Experiment, allow_negative_exponents: bool = False) -> dict:
    """Fits one Extra-P model per (callpath, metric) of an experiment.

    Args:
        experiment: the experiment, see build_experiment.
        allow_negative_exponents: sets Extra-P's `allow_negative_exponents`.
            Use it when a value shrinks as the rank count grows (e.g. strong
            scaling).

    Returns:
        dict: (Callpath, Metric) -> Extra-P model
    """
    model_generator = ModelGenerator(
        experiment, modeler="Basic" if allow_negative_exponents else "Default"
    )
    if allow_negative_exponents:
        model_generator.modeler.allow_negative_exponents = True
    model_generator.model_all()
    return model_generator.models


def print_models(models: dict) -> None:
    for (callpath, metric), model in models.items():
        fn = model.hypothesis.function
        print(f"{callpath} / {metric}\n  {fn}  (aR2={model.hypothesis.AR2:.3f})")


def main(files):
    """Baseline approach to generate a model reusing ioparse functions"""
    # records from trace
    printer = Print(["ioparse", *files], prepare_plot_data=False)
    records = printer.collect_records()

    # build experiment
    experiment = build_experiment(records)

    print(
        f"\nexperiment: {len(experiment.coordinates)} coordinate(s), "
        f"{len(experiment.measurements)} series, "
        f"parameters={[p.name for p in experiment.parameters]}"
    )

    # build model/ fit
    models = fit_models(experiment)
    print_models(models)

    return models


def run():
    """Console-script entry point: console scripts call their target with no arguments."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
