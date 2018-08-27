def sbatch_script = '''#!/bin/bash -l
#SBATCH --job-name=by_test_job
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=test_job.out

date
module list
'''

stage('Test') {
    def machineName = 'scs_daintvm1'
    node(machineName) {
        checkout scm
        sh("""echo '$sbatch_script' > test_script.sh
              sbatch --wait test_script.sh
              cat test_job.out""")

        deleteDir()
    }
}

