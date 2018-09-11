import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS/11Aug17-CrayGNU-17.08-cuda-8.0-mpip-3.4.1']

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

        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task)
                }

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

        self.job.launcher = LauncherWrapper(self.job.launcher, '/usr/bin/time',
                                            ['-p'])


class LAMMPSmpiP(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s_mpip' % (str(num_tasks)), **kwargs)

        input_name = "-in in.lj_12"
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


def _get_checks(**kwargs):
    ret = []
    for num_tasks in [12, 24, 48, 96, 192, 384]:
        ret.append(LAMMPSmpiP(num_tasks, **kwargs))

    return ret

# > ~/reframe.git/reframe.py --system dom:gpu --exec-policy async --keep-stage-files --prefix=$SCRATCH/reframe -r -c ./lammps_mpip.py
#
# Command line: /users/piccinal/reframe.git/reframe.py --system dom:gpu --exec-policy async --keep-stage-files --prefix=/scratch/snx1600tds/piccinal/reframe -r -c ./lammps_mpip.py
# Reframe version: 2.12-dev2
# Launched by user: piccinal
# Launched on host: dom101
# Reframe paths
# =============
#     Check prefix      :
#     Check search path : './lammps_mpip.py'
#     Stage dir prefix  : /scratch/snx1600tds/piccinal/reframe/stage/
#     Output dir prefix : /scratch/snx1600tds/piccinal/reframe/output/
#     Logging dir       : /scratch/snx1600tds/piccinal/reframe/logs
# [==========] Running 2 check(s)
# [==========] Started on Wed Apr 18 10:45:03 2018
#
# [----------] started processing LAMMPS_24 (LAMMPS PerfHackathon benchmark)
# [ RUN      ] LAMMPS_24 on dom:gpu using PrgEnv-gnu
# [----------] finished processing LAMMPS_24 (LAMMPS PerfHackathon benchmark)
#
# [----------] started processing LAMMPS_48 (LAMMPS PerfHackathon benchmark)
# [ RUN      ] LAMMPS_48 on dom:gpu using PrgEnv-gnu
# [----------] finished processing LAMMPS_48 (LAMMPS PerfHackathon benchmark)
#
# [----------] waiting for spawned checks to finish
# [       OK ] LAMMPS_24 on dom:gpu using PrgEnv-gnu
# [       OK ] LAMMPS_48 on dom:gpu using PrgEnv-gnu
# [----------] all spawned checks have finished
# [  PASSED  ] Ran 2 test case(s) from 2 check(s) (0 failure(s))
# [==========] Finished on Wed Apr 18 10:45:44 2018
#
