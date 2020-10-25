# flood_risk_shipston



---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).

> conda env create -f environment.yml


> conda activate ship-evt3

Had to Soften python=3.8 requirement.
====================================


Package gmp conflicts for:
scipy -> libgcc -> gmp[version='>=4.2']
rpy2=2.9.4 -> libgcc -> gmp[version='>=4.2']

Package certifi conflicts for:
pip -> setuptools -> certifi[version='>=2016.09|>=2016.9.26']
matplotlib -> matplotlib-base[version='>=3.3.2,<3.3.3.0a0'] -> certifi[version='>=2016.09|>=2016.9.26|>=2020.06.20']

Package libgfortran4 conflicts for:
scipy -> libgfortran4[version='>=7.5.0']
scipy -> libgfortran=4 -> libgfortran4
r -> r-base[version='>=4.0,<4.1.0a0'] -> libgfortran4[version='>=7.5.0']

Package six conflicts for:
pandas -> python-dateutil[version='>=2.7.3'] -> six[version='>=1.5']
scipy -> mkl-service[version='>=2,<3.0a0'] -> six
rpy2=2.9.4 -> six
matplotlib -> cycler -> six[version='>=1.5']

Package llvm-openmp conflicts for:
r -> r-base[version='>=4.0,<4.1.0a0'] -> llvm-openmp[version='>=10.0.0|>=10.0.1|>=11.0.0|>=9.0.1|>=9.0.0|>=8.0.1|>=8.0.0|>=4.0.1']
rpy2=2.9.4 -> r-base[version='>=3.5,<3.6.0a0'] -> llvm-openmp[version='10.0.0.*|>=10.0.0|>=4.0.1|>=8.0.0|>=8.0.1|>=9.0.1|>=9.0.0']
scipy -> libgfortran=4 -> llvm-openmp[version='>=10.0.0|>=10.0.1|>=8.0.0|>=8.0.1|>=11.0.0|>=9.0.1|>=9.0.0']

Package wheel conflicts for:
python=3.8 -> pip -> wheel
pip -> wheel

Package ca-certificates conflicts for:
pandas -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates
matplotlib -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates
python=3.8 -> openssl[version='>=1.1.1h,<1.1.2a'] -> ca-certificates
pip -> python -> ca-certificates
scipy -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates
jupyter -> python -> ca-certificates

Package libopenblas conflicts for:
scipy -> libblas[version='>=3.8.0,<4.0a0'] -> libopenblas[version='0.3.3|0.3.3|0.3.3|>=0.3.10,<0.3.11.0a0|>=0.3.10,<1.0a0|>=0.3.9,<0.3.10.0a0|>=0.3.9,<1.0a0|>=0.3.8,<0.3.9.0a0|>=0.3.8,<1.0a0|>=0.3.7,<0.3.8.0a0|>=0.3.7,<1.0a0|>=0.3.6,<0.3.7.0a0|>=0.3.6,<1.0a0',build='hdc02c5d_1|hdc02c5d_3|hdc02c5d_2']
pandas -> numpy[version='>=1.18.5,<2.0a0'] -> libopenblas[version='>=0.2.20,<0.2.21.0a0|>=0.3.2,<0.3.3.0a0|>=0.3.3,<1.0a0']
rpy2=2.9.4 -> r-base[version='>=3.5,<3.6.0a0'] -> libopenblas[version='>=0.2.20,<0.2.21.0a0']
matplotlib -> numpy=1.11 -> libopenblas[version='>=0.2.20,<0.2.21.0a0|>=0.3.2,<0.3.3.0a0|>=0.3.3,<1.0a0']
scipy -> libopenblas[version='>=0.2.20,<0.2.21.0a0|>=0.3.2,<0.3.3.0a0|>=0.3.3,<1.0a0']
r -> r-base[version='>=3.5,<3.6.0a0'] -> libopenblas[version='>=0.2.20,<0.2.21.0a0']

Package python conflicts for:
jupyter -> ipykernel -> python[version='3.4.*|>=3|>=3.4|>=3.9,<3.10.0a0|>=3.6|>=3.5']
scipy -> python_abi=3.9[build=*_cp39] -> python[version='3.7.*|3.8.*|3.9.*']
rpy2=2.9.4 -> jinja2 -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=2.7,<2.8.0a0|>=3.8,<3.9.0a0']
rpy2=2.9.4 -> python[version='>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0']
python=3.8
jupyter -> python[version='2.7.*|3.5.*|3.6.*|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
pip -> setuptools -> python[version='>=3.9,<3.10.0a0']
matplotlib -> python[version='2.7.*|3.4.*|3.5.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0|>=3.9,<3.10.0a0|>=3.5,<3.6.0a0']
pip -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.5,<3.6.0a0|3.4.*']
scipy -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.9,<3.10.0a0|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|3.4.*']
matplotlib -> python_abi=3.8[build=*_cp38] -> python[version='3.6.*|3.7.*|3.8.*|3.9.*|<3|>=3']

Package setuptools conflicts for:
python=3.8 -> pip -> setuptools
pip -> setuptools
matplotlib -> setuptools
rpy2=2.9.4 -> jinja2 -> setuptools

Package freetype conflicts for:
matplotlib -> matplotlib-base[version='>=3.3.1,<3.3.2.0a0'] -> freetype[version='>=2.10.2,<3.0a0']
matplotlib -> freetype[version='2.6.*|>=2.9.1,<3.0a0|>=2.8,<2.9.0a0']

Package libpng conflicts for:
matplotlib -> libpng[version='>=1.6.23,<1.7|>=1.6.37,<1.7.0a0|>=1.6.36,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.34,<1.7.0a0|>=1.6.32,<1.7.0a0']
rpy2=2.9.4 -> r-base[version='>=3.5,<3.6.0a0'] -> libpng[version='>=1.6.22,<1.6.31|>=1.6.32,<1.6.35|>=1.6.34,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.37,<1.7.0a0|>=1.6.28,<1.7|>=1.6.27,<1.7']
r -> r-base[version='>=4.0,<4.1.0a0'] -> libpng[version='1.6.*|>=1.6.22,<1.6.31|>=1.6.32,<1.6.35|>=1.6.34,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.37,<1.7.0a0|>=1.6.28,<1.7|>=1.6.27,<1.7|>=1.6.32,<1.7.0a0']
r -> libpng
matplotlib -> freetype=2.6 -> libpng[version='1.6.*|>=1.6.21,<1.7|>=1.6.32,<1.6.35']

Package pypy3.6 conflicts for:
pandas -> pypy3.6[version='>=7.3.1|>=7.3.2']
pandas -> python[version='>=3.6,<3.7.0a0'] -> pypy3.6[version='7.3.*|7.3.0.*|7.3.1.*|7.3.2.*']

Package r conflicts for:
rpy2=2.9.4 -> r-rsqlite -> r[version='3.1.3.*|3.2.0.*|3.2.1.*|3.2.2.*']
r

Package tzdata conflicts for:
scipy -> python[version='>=3.9,<3.10.0a0'] -> tzdata
pip -> python[version='>=3'] -> tzdata
pandas -> python[version='>=3.9,<3.10.0a0'] -> tzdata
jupyter -> python -> tzdata
matplotlib -> python[version='>=3.9, <3.10.0a0'] -> tzdata

Package python-dateutil conflicts for:
pandas -> python-dateutil[version='>=2.5.*|>=2.6.1|>=2.7.3']
matplotlib -> matplotlib-base[version='>=3.3.2,<3.3.3.0a0'] -> python-dateutil[version='>=2.1']
matplotlib -> python-dateutil

Package libgfortran5 conflicts for:
scipy -> libgfortran=5 -> libgfortran5
scipy -> libgfortran5[version='>=9.3.0']

Package tornado conflicts for:
matplotlib -> tornado
jupyter -> ipykernel -> tornado[version='>=4|>=4,<6|>=4.0|>=4.2|>=5.0|>=5.0,<7|>=4.1,<7']

Package libcxx conflicts for:
pandas -> libcxx[version='>=10.0.0|>=10.0.1|>=9.0.1|>=9.0.0|>=4.0.1']
pandas -> python[version='>=3.6,<3.7.0a0'] -> libcxx

Package jinja2 conflicts for:
jupyter -> nbconvert -> jinja2[version='>=2.4']
rpy2=2.9.4 -> jinja2
