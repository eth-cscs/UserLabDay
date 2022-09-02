# How to run them on daint

```bash
# Skip this if you have the tests locally
git clone https://github.com/eth-cscs/UserLabDay.git
cd UserLabDay/2022/automating_proposal_submission/tests/

module load reframe-cscs-tests
reframe -c 4_check_horovod_param.py -r --performance-report
```
