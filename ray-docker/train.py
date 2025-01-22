import os
import torch
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
    .env_runners(num_env_runners=1)
)
algo = config.build()
result = algo.train()

path = os.path.join(os.environ['SM_MODEL_DIR'], "result")
torch.save(result, path)
