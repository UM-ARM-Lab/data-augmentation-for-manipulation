import pathlib

import tabulate

from analysis.analyze_results import load_table_specs, reduce_planning_metrics, load_planning_results
from analysis.results_utils import get_all_results_subdirs
from arc_utilities import ros_init
from link_bot_pycommon.pandas_utils import df_where
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


@ros_init.with_ros("scratch")
def main():
    root = pathlib.Path('/media/shared/planning_results')
    results_dirs = get_all_results_subdirs(root)
    df = load_planning_results(results_dirs, regenerate=False)

    tables_config = pathlib.Path("../tables_configs/planning_evaluation.hjson")
    table_format = tabulate.simple_separated_format("\t")
    table_specs = load_table_specs(tables_config, table_format)

    where_car3 = df_where(df, 'classifier_name', 'val_car_new3/June_01_13-03-21_345ca5f528')
    where_car3_to_long_hook = df_where(where_car3, 'target_env', 'long_hook2')
    agg = where_car3_to_long_hook.groupby(["seed", "stop_on_error", "timeout", "recovery_name", "results_folder_name", 'target_env']).agg(['mean'])
    agg = where_car3_to_long_hook.groupby(["seed", "stop_on_error", "timeout", "recovery_name", "results_folder_name", 'target_env']).size()

    for spec in table_specs:
        data_for_table = reduce_planning_metrics(spec.reductions, df)
        spec.table.make_table(data_for_table)
        spec.table.print()
        print()
        print()


if __name__ == '__main__':
    main()
