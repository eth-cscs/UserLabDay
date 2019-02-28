import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS']

        # Reset sources dir relative to the SCS apps prefix
        #self.sourcesdir = os.path.join(self.current_system.resourcesdir,
        #                               'LAMMPS')
        self.sanity_patterns = sn.assert_found(r'Total wall time:',
                                               self.stdout)

        self.perf_patterns = {
            'perf': sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                     self.stdout, 'perf', float),
        }

        self.maintainers = ['TR', 'VH']
        self.strict_check = False
        self.tags = {'scs'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'LAMMPS PerfHackathon benchmark'

        self.executable = 'lmp_mpi'

        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1

        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task)
                }

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

        self.job.launcher = LauncherWrapper(self.job.launcher, '/usr/bin/time',
                                            ['-p'])


class LAMMPSStrongScaling(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s_strong' % (str(num_tasks)), **kwargs)

        input_name = "-in in.lj_12"
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


class LAMMPSWeakScaling(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s_weak' % (str(num_tasks)), **kwargs)

        input_name = "-in in.lj_%s" % (str(num_tasks))
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


def _get_checks(**kwargs):
    ret = []
    for num_tasks in [12, 24, 48, 96, 192, 384]:
        ret.append(LAMMPSStrongScaling(num_tasks, **kwargs))
        ret.append(LAMMPSWeakScaling(num_tasks, **kwargs))

    return ret
