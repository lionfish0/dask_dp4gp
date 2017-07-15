# dask_dp4gp
A small repo to allow the distribution of dp4gp using DASK

# Installation

Step 1: Install DASK etc (follow instructions in https://github.com/lionfish0/dp/blob/master/Dask.ipynb)

Step 2: Launch DASK server:

```dask-ec2 up --keyname research --keypair .ssh/research.pem --region-name eu-west-1 --ami ami-d37961b5 --tags research:dp --count 4 --volume-size 30 --type c4.8xlarge```

Step 3: On the scheduler, we're going to awkwardly install this repo twice, once to get easy access to the jupyter notebooks and once to get easy access for import... maybe I should split the notebooks into yet another repo!

```git clone https://github.com/lionfish0/dask_dp4gp.git```

```pip install git+https://github.com/lionfish0/dask_dp4gp.git```

Once done visit the jupyter notebooks for further instructions.
