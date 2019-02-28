import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS/16Jul2018-CrayGNU-18.07-cuda-9.1-ddt']
        self.modules += ['ddt/18.1.3-Suse-12']
        self.sourcesdir = 'src/sph/water_collapse/010000wp'

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
        self.time_limit = (0, 20, 0)
        self.pre_run = ['gunzip data.initial.gz']
        self.pre_run += ['sed -i "s/equal 10000/equal 20/" water_collapse.lmp']
        self.pre_run += ['grep "^variable           nrun equal" water_collapse.lmp']
        self.post_run = ['module list -t']
        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task)
                }

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

		# src/pair_sph_taitwater.cpp:
		# 193         if (evflag)
		# 194            ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
        self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt --connect --break-at pair_sph_taitwater.cpp:194 --mem-debug')
        self.job.launcher.options = ['--unbuffered']


class LAMMPSStrongScaling(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s_100000wp_strong+ddt_connect+mem' % (str(num_tasks)), **kwargs)

        input_name = '-in water_collapse.lmp'
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


def _get_checks(**kwargs):
    ret = []
    for num_tasks in [12]:
        ret.append(LAMMPSStrongScaling(num_tasks, **kwargs))

    return ret
