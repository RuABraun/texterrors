from distutils.core import setup, Extension

setup(
    name='textalign',
    version='0.1',
    packages='textalign',
    license='',
    long_description='BLABLA',
    ext_modules=[Extension('textalign.fast', ['libs/fast.cpp'])],
)
