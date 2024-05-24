from setuptools import setup

setup(
    name='spectral_extraction',
    version='0.0.1',    
    description="LITTLE CODE TO EXTRACT MULTIPLE 1D ESPECTRA FROM ONE 2D",
    url='https://github.com/felavila/code_for_extraction_2d_spectra',
    author='Felipe Avila-Vera',
    author_email='shudson@anl.gov',
    license='MIT License',
    packages=['spectral_extraction'],
    install_requires=['astropy>=6.0', "numpy>=1.26.3",                   
                      "lmfit>=1.2.2","scipy>=1.11.4","pandas>=2.2.0","matplotlib>=3.8.2",
                      "parallelbar>=2.4"],
     classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
         'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.10',
    ],
)