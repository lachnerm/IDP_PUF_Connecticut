import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

parser = argparse.ArgumentParser()
parser.add_argument('--conc_tasks', type=int, default=2)
parser_args = parser.parse_args()

bs_values = [16, 32]
lr_values = [0.0005, 0.001, 0.002]
ns_values = [64, 96, 128]
beta_values = [0.3, 0.5, 0.7, 0.9, 0.999]
network_values = [True, False]
act_values = ["LeakyReLU", "GELU", "ELU"]

hparams = [
    DiscreteParameterRange('General/bs', values=bs_values),
    DiscreteParameterRange('General/lr', values=lr_values),
    DiscreteParameterRange('General/ns', values=ns_values),
    DiscreteParameterRange('General/beta1', values=beta_values),
    DiscreteParameterRange('General/beta2', values=beta_values),
    DiscreteParameterRange('General/new_network', values=network_values),
    DiscreteParameterRange('General/act', values=act_values),
]

aSearchStrategy = OptimizerOptuna


def job_complete_callback(job_id, objective_value, objective_iteration, job_parameters, top_performance_job_id):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print(f'Objective reached {objective_value}')


task = Task.init(project_name='Nonlinear PUF 2048 Hyper-Parameter Optimization 2',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

args = {
    'template_task_id': None,
    'run_as_service': False,
}

args = task.connect(args)

if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(project_name='Nonlinear PUF 2048', task_name="PUF 2048").id

execution_queue = 'hparam'

an_optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=hparams,
    objective_metric_title='Validation Average Pearson Correlation',
    objective_metric_series='Validation Average Pearson Correlation',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=parser_args.conc_tasks,
    optimizer_class=aSearchStrategy,
    execution_queue=execution_queue,
    spawn_project=None,
    save_top_k_tasks_only=None,
    time_limit_per_job=30.,
    pool_period_min=3.,
    total_max_jobs=None,
    min_iteration_per_job=30,
    max_iteration_per_job=300,
)

if args['run_as_service']:
    task.execute_remotely(queue_name='services', exit_process=True)

an_optimizer.set_report_period(2.2)
an_optimizer.start(job_complete_callback=job_complete_callback)
an_optimizer.set_time_limit(in_minutes=2000.)
an_optimizer.wait()

top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
an_optimizer.stop()

print('Hparam search finished.')
