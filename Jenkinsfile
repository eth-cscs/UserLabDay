stage('Test') {
    def machineName = 'scs_daintvm1'
    node(machineName) {
        checkout scm
        sh("""sbatch --wait test_script.sh
              cat test_job.out""")

        archiveArtifacts '*.err,*.out'
        deleteDir()
    }
}

