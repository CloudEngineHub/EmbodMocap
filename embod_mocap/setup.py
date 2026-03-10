from setuptools import setup, find_packages

setup(
    name="embod_mocap",
    version="0.1.0", 
    description="A project with multiple packages for training, modeling, and data processing.", 
    author="Wenjia Wang",
    author_email="wwj2022@connect.hku.hk",
    license="Apache-2.0",
    url="https://github.com/your-repo/my_project", 
    packages=find_packages(
        where=".",  
    ),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)