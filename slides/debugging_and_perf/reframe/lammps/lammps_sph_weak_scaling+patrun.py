import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS/16Jul2018-CrayGNU-18.07-cuda-9.1']
        self.sourcesdir = 'src/sph/water_collapse'
        self.sourcepath_weak = {
            12: os.path.join(self.sourcesdir,'100000wp'),
            24: os.path.join(self.sourcesdir,'200000wp'),
            48: os.path.join(self.sourcesdir,'400000wp'),
            96: os.path.join(self.sourcesdir,'800000wp'),
        }

        self.sanity_patterns = sn.assert_found(r'Total wall time:',
                                               self.stdout)

        self.perf_patterns = {
            'perf': sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                     self.stdout, 'perf', float),
        }

        self.maintainers = ['JG', 'MKr']
        self.strict_check = False
        self.tags = {'scs'}
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'LAMMPS PerfHackathon benchmark'
        self.executable = 'pat_run lmp_mpi'
        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1
        self.time_limit = (0, 20, 0)
        self.pre_run = ['gunzip data.initial.gz']
        self.post_run = ['module list -t ;pat_report lmp_mpi+*s &> xf.rpt ;dos2unix xf.rpt']
        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task)
                }

    def setup(self, system, environ, **job_opts):
        self.sourcesdir = self.sourcepath_weak[self.num_tasks]
        super().setup(system, environ, **job_opts)

        self.job.launcher = LauncherWrapper(self.job.launcher, '/usr/bin/time',
                                            ['-p'])
        #sbatch: self.job.options = ['--unbuffered']
        #srun:
        self.job.launcher.options = ['--unbuffered']


class LAMMPSWeakScaling(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):

        super().__init__('LAMMPS_%s_weak+patrun' % (str(num_tasks)), **kwargs)

        input_name = '-in water_collapse.lmp'
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


def _get_checks(**kwargs):
    ret = []
    for num_tasks in [12, 24, 48, 96     ]:
        ret.append(LAMMPSWeakScaling(num_tasks, **kwargs))

    return ret
