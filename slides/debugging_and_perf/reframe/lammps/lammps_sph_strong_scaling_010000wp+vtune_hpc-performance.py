import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSPerfHackathon(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['LAMMPS/16Jul2018-CrayIntel-18.07']
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
        #self.valid_systems = ['daint:mc', 'dom:mc']
        self.descr = 'LAMMPS PerfHackathon benchmark'
        #self.executable = 'lmp_mpi'
        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1
        self.time_limit = (0, 10, 0)
        self.pre_run = ['gunzip data.initial.gz']
        self.pre_run += ['module list -t && ldd `which lmp_mpi` && ldd `which lmp_mpi` && ']
        self.pre_run += ['XX=/apps/common/UES/intel/2019 && VV=$XX/vtune_amplifier && ' \
                         'source $VV/amplxe-vars.sh && amplxe-cl -help collect ;']
        #self.post_run = ['module list -t']
        self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task)
                }

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

        self.job.launcher = LauncherWrapper(self.job.launcher, '/usr/bin/time',
                                            ['-p'])
        self.job.launcher.options = ['--unbuffered']


class LAMMPSStrongScaling(LAMMPSPerfHackathon):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('LAMMPS_%s_010000wp_strong+vtune_hpc' % (str(num_tasks)), **kwargs)

        self.executable = 'amplxe-cl -collect hpc-performance -trace-mpi -r $SCRATCH/lammps/vtune/hpc/%s -data-limit=0 lmp_mpi' \
                % (str(num_tasks))
        input_name = '-in water_collapse.lmp'
        self.executable_opts = [input_name]
        self.num_tasks = num_tasks


def _get_checks(**kwargs):
    ret = []
    #for num_tasks in [12, 24, 48, 96, 192]:
    #for num_tasks in [1,2,4,8,12]:
    for num_tasks in [12]:
        ret.append(LAMMPSStrongScaling(num_tasks, **kwargs))

    return ret
