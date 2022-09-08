jaisalab: A Toolkit for Safe RL
-------------------------------

jaisalab is a framework for developing and evaluating reinforcement learning algorithms that take into account constraints that should be satisfied in OpenAI gym environments, that also provides PyTorch implementations for a few state-of-the-art algorithms in Constrained RL. 
Namely, the algorithms implemented are [Constrained Policy Optimization (CPO)](https://github.com/jaimesabalimperial/jaisalab/blob/master/jaisalab/algos/cpo.py), the [Safety Augmented RL (SAUTE)](https://github.com/jaimesabalimperial/jaisalab/blob/master/jaisalab/envs/saute_env.py) wrapper that incorporates safety to any RL algorithm in a plug-n-play manner, and **Distributional Constrained Policy Optimization**, a modification to CPO that exploits uncertainty in the cost value function to better satisfy constraints. 

The framework builds on the [garage](https://github.com/rlworkgroup/garage) toolkit to add functionality to safe RL settings (i.e. where there is a cost associated with each state-action pair on top of the reward) and thus provides an auxiliary set of modular tools for implementing constrained RL algorithms, analogously to how **garage** provides these for regular RL settings. 

Running Unit and Integration Tests
----------------------------------

This project uses pytest for unit and integration testing (included in the 
developer dependencies). The tests may be run from the root directory as 
follows:

.. code-block:: console

    $ pytest
    ...
    ===== x passed, x warnings in x.xx seconds =====


Instructions for Experiment Reproduction
----------------------------------------

The final results for DCPO, Saute TRPO, Vanilla TRPO, and CPO showed in the final report are equivalent to the experiments in [jaisalab.experiments.backlog](https://github.com/jaimesabalimperial/jaisalab/blob/master/jaisalab/experiments/backlog.py), ran across seeds 1-10. The evaluation of the trained algorithms is done through the SeedEvaluator object for 30 epochs and the plots are made using the methods of the RLPlotter object. 

References
----------

The code implementations for the algorithms present in **jaisalab** are based partly or fully on the repositories of their respective papers. For **CPO**, the original TensorFlow implementation can be found in [here](https://github.com/jachiam/cpo), but this had to be translated to PyTorch to fit our framework. For **SAUTE**, the implementation was much more straightforward since it's simply a wrapper around the environment, rather than a separate algorithm that should fit our framework as a whole, so the code present in **jaisalab/envs/safe_env** and **jaisalab/envs/saute_env** was taken directly from [Huawei's repository](https://github.com/huawei-noah/HEBO) (more specifically [here](https://github.com/huawei-noah/HEBO/tree/405dc4ceb93a79f0d1f0eaa24f5458dd26de1d05/SAUTE/envs/wrappers)). 

Citing jaisalab
---------------

If you use jaisalab for academic research, please cite the repository using the
following BibTeX entry. You should update the `commit` field with the commit or
release tag your publication uses.

```latex
@misc{jaisalab,
 author = {Jaime Sabal Bermúdez},
 title = {jaisalab: A garage-based framework for reproducible constrained reinforcement learning research},
 year = {2022},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/jaimesabalimperial/jaisalab}},
 commit = {cc5bc6b4dc7074af1f47d21c6d312429b2ccb931}
}
```

Notes
-----

Developed in partial fulfillment of the individual project of Jaime Sabal Bermúdez for the MSc degree in Artificial Intelligence of Imperial College London. 

I would like to thank Dr. Calvin Tsay for his guidance and advice throughout the project. 

**Disclaimer:** While jaisalab is an open-source project and contributions are 
welcome, we cannot guarantee that the codebase will be actively maintained in 
the future. 

