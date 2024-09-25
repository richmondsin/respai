from setuptools import setup, find_packages

setup(
    name='off_topic_detector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'transformers',
        'torch',
        'tqdm',
    ],
    include_package_data=True,
    package_data={
        'off_topic_detector': ['models/*.pkl'],
    },
    author='Richmond Sin',
    author_email='richmondsin.rs@gmail.com',
    description='A tool to detect off-topic prompts for LLM-powered chatbots.'
)