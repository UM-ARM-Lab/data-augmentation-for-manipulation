#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import tabulate

from analysis.analyze_results import load_table_specs, reduce_planning_metrics, load_planning_results
from analysis.results_metrics import load_analysis_params
from analysis.results_utils import get_all_results_subdirs
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    analysis_params = load_analysis_params(args.analysis_params)

    # The default for where we write results
    out_dir = args.results_dirs[0]

    print(f"Writing analysis to {out_dir}")

    if args.latex:
        table_format = 'latex_raw'
    else:
        table_format = tabulate.simple_separated_format("\t")

    results_dirs = get_all_results_subdirs(args.results_dirs)
    df = load_planning_results(results_dirs, regenerate=args.regenerate)
    df.to_csv("/media/shared/analysis/all_results.csv")

    # Figures & Tables
    # figspecs = load_fig_specs(analysis_params, args.figures_config)
    table_specs = load_table_specs(args.tables_config, table_format)

    # z = df.copy()
    # z.set_index(['classifier_source_env', 'target_env'], inplace=True)
    # plt.figure()
    # df.melt(id_vars=["subidr", "attnr"], var_name="solutions", value_name="score").
    # sns.boxplot(
    #     x=['classifier_source_env', 'target_env'],
    #     y="task_error",
    #     hue=('classifier_source_env', 'target_env'),
    #     data=z
    # )

    # for spec in figspecs:
    #     data_for_figure = get_data_for_figure(spec, df)
    #
    #     # spec.fig.make_figure(data_for_figure, method_names)
    #     # spec.fig.save_figure(out_dir)

    for spec in table_specs:
        data_for_table = reduce_planning_metrics(spec.reductions, df)
        spec.table.make_table(data_for_table)
        spec.table.save(out_dir)

    for spec in table_specs:
        print()
        spec.table.print()
        print()

    tables_outfilename = out_dir / 'tables.txt'
    with tables_outfilename.open("w") as tables_outfile:
        for spec in table_specs:
            tables_outfile.write(spec.table.table)
            tables_outfile.write('\n\n\n')

    # if not args.no_plot:
    #     for spec in figspecs:
    #         spec.fig.fig.set_tight_layout(True)
    #     plt.show()


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--figures-config', type=pathlib.Path,
                        default=pathlib.Path("figures_configs/planning_evaluation.hjson"))
    parser.add_argument('--tables-config', type=pathlib.Path,
                        default=pathlib.Path("tables_configs/planning_evaluation.hjson"))
    parser.add_argument('--analysis-params', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/env_across_methods.json"))
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('--order', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--debug', action='store_true', help='will only run on a few examples to speed up debugging')
    parser.add_argument('--style', default='slides')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)

    metrics_main(args)


if __name__ == '__main__':
    main()
