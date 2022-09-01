# Finetuning BERT on SQuaD for the Question and Answering task

In this example, we are going to fine-tune [BERT implemented by HuggingFace](https://huggingface.co/bert-base-uncased) for the text-extraction task with the [SQuAD (The Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/) dataset. The data is composed by a set of questions and the corresponding paragraphs that contain the answers. The model will be finetuned to locate the answer in the context by giving the positions where the answer starts and ends.

For that we will use the PyTorch installation from the images provided by [Nvidia GPU Cloud](https://catalog.ngc.nvidia.com/).

## Get the image
Choose your preferred image [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) and pull it to the sarus database on Piz Daint (the `sarus` module needs to be loaded - `module load sarus`):
```bash
sarus pull nvcr.io/nvidia/pytorch:22.08-py3
```

## Prepare the software
Not all the packages that we need are available in the container. For this example, we need to install `deepspeed`, HuggingFace's `datasets`, `tokenizers` and `transformers`, `MPI4Py` and `rich`. We will install them within the container, but using a host path so the packages won't go away after the container exits. To do that, let's run the container interactively with the `$SCRATCH` directory mounted and create there a virtual environment:
```bash
sarus run --tty --mount=type=bind,source=$SCRATCH,destination=$SCRATCH \
          nvcr.io/nvidia/pytorch:22.08-py3 bash
cd $SCRATCH/UserLabDay/2022/containers/ngc-bert-squad
python -m venv deepspeed-env --system-site-packages
. deepspeed-env/bin/activate
pip install deepspeed
pip install datasets tokenizers transformers
MPICC=mpicc pip install mpi4py
pip install rich
ln -s /opt/conda/lib/libxgboost.so $SCRATCH/UserLabDay/2022/containers/ngc-bert-squad/deepspeed-env/lib/libxgboost.so
```
The symlink in the last line needs to be done because in the `deepspeed-env`,  `xgboost` can't find the `libxgboost.so` library where it's originally installed.

## Running
Now we are ready to run our script.

On Piz Daint, we use Slurm's `srun` to launch executables within an allocation. That won't work with this container because the software that comes with it has been built with OpenMPI, which in turn, wasn't build with Slurm support. So, launching our script with `srun` crashes with an error like this: 
```text
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support and therefore cannot
execute. There are several options for building PMI support under
SLURM, depending upon the SLURM version you are using:

...
```
Fortunately, with deep learning applications like PyTorch and TensorFlow, all the communication is typically done via [NCCL](https://github.com/NVIDIA/nccl). This means that the only thing needed from OpenMPI is the `mpirun` command. We can use the `mpirun` of any host installation of OpenMPI to launch the container. OpenMPI can be installed with EasyBuild using [this recipe](https://github.com/eth-cscs/production/blob/master/easybuild/easyconfigs/o/OpenMPI/OpenMPI-4.1.2.eb) (more info [here](https://user.cscs.ch/computing/compilation/easybuild/)).

We can use the Slurm [`run.sh`](run.sh) script to run the example.

The output of DeepSpeed is large. During training you will see something like this
```
[2022-08-29 06:58:56,121] [INFO] [logging.py:68:log_dist] [Rank 0] step=20, skipped=17, lr=[5e-05], mom=[(0.9, 0.999)]
[2022-08-29 06:58:56,176] [INFO] [timer.py:198:stop] 0/20, RunningAvgSamplesPerSec=232.7340551694273, CurrSamplesPerSec=227.48090316379353, MemAllocated=0.36GB, MaxMemAllocated=15.06GB
[2022-08-29 06:59:18,633] [INFO] [logging.py:68:log_dist] [Rank 0] step=30, skipped=17, lr=[5e-05], mom=[(0.9, 0.999)]
[2022-08-29 06:59:18,687] [INFO] [timer.py:198:stop] 0/30, RunningAvgSamplesPerSec=230.82135336499695, CurrSamplesPerSec=227.45044884073528, MemAllocated=0.36GB, MaxMemAllocated=15.06GB
[2022-08-29 06:59:41,140] [INFO] [logging.py:68:log_dist] [Rank 0] step=40, skipped=17, lr=[5e-05], mom=[(0.9, 0.999)]
[2022-08-29 06:59:41,193] [INFO] [timer.py:198:stop] 0/40, RunningAvgSamplesPerSec=229.93929267351086, CurrSamplesPerSec=227.7217889790298, MemAllocated=0.36GB, MaxMemAllocated=15.06GB
```

When the training finishes, the finetuned model will be saved and examples where the model highlights the answer will be printed.

## Testing the model
The model can be tried on the test set with the [`test.sh`](test.sh) script. There we add `--tty` to `sarus run` so the output is printed formatted with colors.
