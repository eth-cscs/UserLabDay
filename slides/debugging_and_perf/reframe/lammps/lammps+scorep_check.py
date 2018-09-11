import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']

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

        self.executable = 'lmp_mpi'
        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1

        self.cube_file = 'cube.rpt'
        self.post_run = [
            'scorep-score -r -m scorep-*/profile.cubex > %s' % self.cube_file
        ]

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

        self.job.launcher = LauncherWrapper(self.job.launcher, '/usr/bin/time',
                                            ['-p'])


class LAMMPS_ScoreP_Profile(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s' % (str(num_tasks)), **kwargs)

        self.modules = ['LAMMPS/11Aug17-CrayGNU-17.08-cuda-8.0-scorep-3.1']

        input_name = "-in in.lj_12"
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks

        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                'SCOREP_ENABLE_PROFILING': 'true',
                'SCOREP_ENABLE_TRACING': 'false',
                'SCOREP_TIMER': 'clock_gettime'
                }


# THIS IS MORE FOR DOCUMENTATION
#
# class LAMMPS_ScoreP_Filter(LAMMPSPerfHackathon):
#     def __init__(self, num_tasks, **kwargs):
#         super().__init__('LAMMPS_%s' % (str(num_tasks)), **kwargs)
#
#         self.modules = ['LAMMPS/11Aug17-CrayGNU-17.08-cuda-8.0-scorep-3.1']
#
#         input_name = "-in in.lj_12"
#         self.executable_opts = [input_name]
#         self.num_tasks = num_tasks
#
#         self.variables = {
#                 'OMP_NUM_THREADS': str(self.num_cpus_per_task),
#                 'SCOREP_ENABLE_PROFILING': 'false',
#                 'SCOREP_ENABLE_TRACING': 'true',
#                 'SCOREP_FILTERING_FILE': 'my_filter',
#                 'SCOREP_TIMER': 'clock_gettime'
#                 }
#
#         self.cube_file = 'cube_filtered'
#         self.pre_run = [
#                 'scorep-score -f my_filter -r -m scorep-*/profile.cubex > %s'
#                 % self.cube_file
#                 ]


def _get_checks(**kwargs):
    ret = []
    for num_tasks in [24, 48, 96]:
        ret.append(LAMMPS_ScoreP_Profile(num_tasks, **kwargs))

    return ret
