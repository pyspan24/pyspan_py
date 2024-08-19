from setuptools import setup

setup(
    name="pyspan",
    version="0.1.0",
    py_modules=["pyspan"],
    description="A Python package for efficient data cleaning and preprocessing with Pandas.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Noor Surani, Amynah Reimoo",
    author_email="nsurani@hotmail.com, amynahreimoo@gmail.com",
    url="https://github.com/pyspan24/pyspan_py.git",
    install_requires=[
       "numpy==2.0.1",
       "pandas==2.2.2",
       "pyspellchecker==0.8.1",
       "python-dateutil==2.9.0.post0",
       "pytz==2024.1",
       "setuptools==72.2.0",
       "six==1.16.0",
       "tzdata==2024.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
