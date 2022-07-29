import gym

from stable_baselines3 import PPO
from bspointmass import PointEnv

env=  PointEnv()#gym.make("CartPole-v1")#PointEnv
# #
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1E)
# model.save("ppo_Point2")

# del model # remove to demonstrate saving and loading
k=2000
i=0
model = PPO.load("ppo_Point2")

obs = env.reset()
dones=False
while i<k:
    i=i+1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()