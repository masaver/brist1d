
conda create --yes --name=brist1d python=3.12.2
conda install --yes --name brist1d --file requirements_conda.txt
conda run --name brist1d pip3 install -r requirements_pip.txt

# Clone pyenv-virtualenv
# git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

#Steps to install py env
Example Workflow

    Install Python 3.9.10:

    bash

pyenv install 3.9.10

Create a virtual environment called myproject-env using Python 3.9.10:

bash

pyenv virtualenv 3.9.10 myproject-env

Set myproject-env as the local environment for a project folder:

bash

cd /path/to/myproject
pyenv local myproject-env

Install packages into this environment:

bash

    pip install -r requirements.txt

pyenv makes managing multiple Python versions and environments simple, allowing you to isolate project dependencies and control Python versions with ease.
