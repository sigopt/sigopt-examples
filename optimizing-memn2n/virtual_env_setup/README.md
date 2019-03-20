# Instructions to create a python virtualenv for running this end to end memory network implementation.

### Install pip3

#### For mac:

`brew install pip3`

#### For Ubuntu:

`sudo apt-get install python3-pip`

### Install virtualenv

`pip3 install virtualenv`

### Create a new virtualenv

`virtualenv -p $(which python3) <LOCATION OF ENVIRONMENT>`

activate:

`source <LOCATION OF ENVIRONMENT>/bin/activate`

example:

`virtualenv -p $(which python3) ./memn2n_tf_python3_venv`
`source ./memn2n_tf_python3_venv/bin/activate`

### Install requirements

`pip3 install -r <PATH TO REQUIREMENTS FILE>`

#### For GPU compatible environment:

`pip install -r [.]/python3_memn2n_tf_env_gpu_requirements.txt`

#### For CPU compatible environment:

`pip install -r [.]/python3_memn2n_tf_env_cpu_requirements.txt`

### Activate virtualenv

`source <LOCATION OF ENVIRONMENT>/bin/activate`

example:

`source ./memn2n_tf_python3_venv/bin/activate`

To deactivate:

`deactivate`
