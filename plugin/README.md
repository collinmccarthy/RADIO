# RADIO Modifications

This document outlines the additions to RADIO we've made to support more generalized fine-tuning, benchmarking and creating new RADIO models. This 'plugin' directory exists to more easily separate
our changes with the official repository so we can easily pull new changes as necessary.

## `ConfigurableRADIOModel`

- See `plugin/radio/configurable_radio_model.py`
- Adds support for explicit kwargs, which is necessary for creating new RADIO models while supporting current models and their pre-trained checkpoints

## `