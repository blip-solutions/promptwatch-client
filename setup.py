import setuptools

with open("src/promptwatch/__init__.py","rt") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                __version__ = line.split("=")[1].strip(" \n\"")

setuptools.setup(name='promptwatch',
                version=__version__,
                description='promptwatch.io python client to trace langchain sessions',
                long_description=open('README.md').read(),
                long_description_content_type='text/markdown',
                author='Juraj Bezdek',
                author_email='juraj.bezdek@blip.solutions',
                url='https://github.com/blip-solutions/promptwatch-client',
                package_dir={"": "src"},
                packages=setuptools.find_packages(where="src"),
                license='MIT License',
                zip_safe=False,
                keywords='promptwatch prompt monitoring',

                classifiers=[
                ],
                python_requires='>=3.8',
                install_requires=[
                    "langchain",
                    "pydantic",
                    "tiktoken",
                    "tqdm",
                ]
                )
