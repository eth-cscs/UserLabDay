import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS/16Jul2018-CrayGNU-18.07-cuda-9.1-scorep-4.0',
                        'Scalasca/2.4-CrayGNU-18.07']
        self.sourcesdir = 'src/sph/water_collapse/100000wp'

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
        self.executable = '`which lmp_mpi`'
        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1
        self.time_limit = (0, 20, 0)
        self.pre_run = ['gunzip data.initial.gz']
        #self.cube_file = 'cube.rpt'
        self.post_run = [
            'module list -t && square -s scorep_*/ ',
        ]
        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                #'SCOREP_ENABLE_PROFILING': 'false',
                #'SCOREP_ENABLE_TRACING': 'true',
                #'SCOREP_TIMER': 'clock_gettime',
                'SCOREP_FILTERING_FILE': 'filter.jg',
                'SCAN_ANALYZE_OPTS': '--time-correct',
                }

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

        self.job.launcher = LauncherWrapper(self.job.launcher, 'scalasca',
                                            ['-analyze'])
        #self.job.launcher.options = ['--unbuffered']


class LAMMPSStrongScaling(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s_100000wp_strong+scalasca_profiling' % (str(num_tasks)), **kwargs)

        input_name = '-in water_collapse.lmp'
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


def _get_checks(**kwargs):
    ret = []
    #for num_tasks in [12, 24, 48, 96, 192]:
    for num_tasks in [12, 24, 48]:
        ret.append(LAMMPSStrongScaling(num_tasks, **kwargs))

    return ret
