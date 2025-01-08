from setuptools import setup, find_packages

setup(
    name="pyspan",
    version="0.4.6",
    packages=find_packages(),
    description="A Python package for efficient data cleaning and preprocessing",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author="Noor Surani, Amynah Reimoo",
    author_email="nsurani@hotmail.com, amynahreimoo@gmail.com",
    license="MIT",
    url="https://github.com/pyspan24/pyspan_py.git",
    install_requires=[
       "numpy<2.0.0",
       "pandas<=2.2.2",
       "pyspellchecker==0.8.1",
       "scikit-learn<=1.6.0",
       "openai>=1.12.0"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
