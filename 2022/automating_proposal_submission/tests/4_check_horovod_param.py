import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class tensorflow_hvd_param(rfm.RunOnlyRegressionTest):
    descr = 'Distributed CNN training with TensorFlow2 and Horovod'
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    num_nodes = parameter([1, 2, 4, 8])
    num_tasks_per_node = 1
    modules = ['Horovod']
    variables = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_IB_HCA': 'ipogif0'
    }
    executable = 'python tf2_hvd_synthetic_benchmark.py'
    executable_opts = ['--batch-size=128', '--model=ResNet50',
                       '--num-iters=10']
    throughput_per_gpu = 200
    strict_check = False

    @run_before('run')
    def setup_scaling(self):
        self.num_tasks = self.num_nodes

    @run_before('performance')
    def set_references(self):
        throughput_total = self.throughput_per_gpu * self.num_tasks
        self.reference = {
            'daint:gpu': {
                'samples_per_sec_per_gpu': (self.throughput_per_gpu,
                                           -0.1, None, 'samples/sec'),
                'samples_per_sec_total': (throughput_total,
                                          -0.1, None, 'samples/sec')
            }
        }

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Using ipogif0', self.stdout),
            sn.assert_found(r'Total img/sec on \d+ GPU\(s\): \S+ \+', self.stdout)
        ])

    @performance_function('samples/sec')
    def samples_per_sec_per_gpu(self):
        return sn.extractsingle(
            r'Img/sec per GPU: (?P<samples_per_sec_per_gpu>\S+) \+',
            self.stdout, 'samples_per_sec_per_gpu', float
        )

    @performance_function('samples/sec')
    def samples_per_sec_total(self):
        return sn.extractsingle(
            r'Total img/sec on \d+ GPU\(s\): (?P<samples_per_sec_total>\S+) \+',
            self.stdout, 'samples_per_sec_total', float
        )
