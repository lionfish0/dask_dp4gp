from distutils.core import setup
setup(
  name = 'dask_dp4gp',
  packages = ['dask_dp4gp'], # this must be the same as the name above
  version = '1.01',
  description = 'Uses DASK to distribute x-val and grid search dp4gp',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/dask_dp4gp.git',
  download_url = 'https://github.com/lionfish0/dask_dp4gp/archive/1.01.tar.gz',
  keywords = ['differential privacy','gaussian processes','dask'],
  classifiers = [],
  install_requires=['dp4gp','GPy','numpy','sklearn','scipy','matplotlib','dask_searchcv','dask','distributed'],
)
