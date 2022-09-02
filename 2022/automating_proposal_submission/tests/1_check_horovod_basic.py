import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class tensorflow_hvd_basic(rfm.RunOnlyRegressionTest):
    descr = 'Distributed CNN training with TensorFlow2 and Horovod'
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    num_tasks = 4
    num_tasks_per_node = 1
    modules = ['Horovod']
    variables = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_IB_HCA': 'ipogif0'
    }
    executable = 'python tf2_hvd_synthetic_benchmark.py'
    executable_opts = ['--batch-size=128', '--model=ResNet50',
                       '--num-iters=10']

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Using ipogif0', self.stdout),
            sn.assert_found(r'Total img/sec on \d+ GPU\(s\): \S+ \+', self.stdout)
        ])

