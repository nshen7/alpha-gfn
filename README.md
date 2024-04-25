# `Alpha-gfn`: Mining formulaic alpha factors with generative flow networks

In this repo, we build an demo application that leverages a deep reinforcement learning framework to mine formulaic alpha factors using generative flow network models (i.e., GFlowNet). We give a brief introduction on the fundamental components of the project by answering the following questions. 

### What are formulaic alpha factors and why search for them?

In quantitative investment, formulaic alpha factors are mathematical expressions or formulas used to identify and potentially exploit patterns or signals in financial data, for example, US stocks in this demo. These factors are typically derived from historical market data and are used to generate investment signals for trading strategies. Here are some common characteristics of formulaic alpha factors:

1. **Quantitative Formulation:** Alpha factors are expressed mathematically as formulas or algorithms that calculate a numerical value based on input data, such as stock prices, trading volumes, financial ratios, or other market indicators.
1. **Signal Generation:** Alpha factors are designed to capture signals or patterns in market data that are believed to be predictive of future price movements or other market dynamics. These signals may indicate opportunities for buying, selling, or holding securities.
1. **Backtesting and Validation:** Formulaic alpha factors are typically tested and validated using historical data to assess their effectiveness in generating positive returns or outperforming the market.
1. **Integration into Trading Strategies:** Alpha factors are often integrated into quantitative trading strategies, where they serve as inputs for decision-making processes, such as portfolio construction, position sizing, risk management, and trade execution.

Examples of formulaic alpha factors include moving averages, momentum indicators, relative strength indexes, price-to-earnings ratios, and various technical and fundamental indicators, etc. **However, in this demo, we only consider technical indicators and our search space consists of daily frequency market data only, such as open and close price.**

### Basic components of reinforcement learning？

Following is a non-exaustive list of important concepts in reinforcement learning, which are used in the project and will be mentioned in the rest of this introduction.

1. **Agent**: The learner or decision-maker that interacts with the environment. It takes actions and receives rewards based on its actions. In our case, it is the 'alpha generator' modeled by GFlowNet.
1. **Environment**: The external system with which the agent interacts. It receives actions from the agent and returns observations and rewards. In our case, it is the stock market.
1. **State**: A snapshot of the environment at a particular time. It contains all the relevant information necessary for decision-making. In our case, it is the sequence of action tokens that forms the the mathematical expression of an alpha factor. (See [State space](#state))
1. **Action**: A decision made by the agent that affects the state of the environment. The set of all possible actions is called the action space.
1. **Reward**: A scalar feedback signal received by the agent from the environment. It indicates how good or bad the action taken by the agent was.  
1. **Policy**: The strategy or rule that the agent uses to select actions based on states. It defines the mapping from states to actions.

See [Methodology](#Methodology) for detailed description of how these components are defined in this project.

### What are GFlowNet models?

A GFlowNet is a trained stochastic policy or generative model, trained such that it samples objects $x$ through a sequence of constructive steps, with probability proportional to a reward function $R(x)$, where $R$ is a non-negative integrable function. After proper training sessions, a GFlowNet is expected to be able to sample a diversity of solutions $x$ that have a high value of $R(x)$. 

### Why do we choose to apply GFlowNet?



## Dataset
sss

You may find the pre-processing steps in `notebooks/preprocess.ipynb`.

**Data source:**
- The set of stock ticks in S&P500 is extracted from https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv.
- The US stock market data is downloaded from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset.

## Methodology 

Implementation of the methodology can be found in `src/` folder. An example training session and some simple analyses can be found in `notebooks/process.ipynb`.


### Action space <a name="action"></a>

### State space <a name="state"></a>

### Reward <a name="reward"></a>

### Loss function

### Feature extraction

### Policy network architecture

## Future Work


## References:
- **Publications:**
    1. Bengio, Emmanuel, et al. "Flow network based generative models for non-iterative diverse candidate generation." Advances in Neural Information Processing Systems 34 (2021): 27381-27394.
    1. Bengio, Yoshua, et al. "Gflownet foundations." Journal of Machine Learning Research 24.210 (2023): 1-55.
    1. Yu, Shuo, et al. "Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.
- **Scripts and websites:**
    1. The GFlowNet Tutorial: https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3
    1. The smiley face tutorial in GFlowNet GitHub repo: https://github.com/GFNOrg/torchgfn/blob/master/tutorials/notebooks/intro_gfn_smiley.ipynb
    1. The AlphaGen framework: https://github.com/RL-MLDM/alphagen/tree/master
- **Special thanks** to ChatGPT for helping to draft this README!!!