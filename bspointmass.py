import gym
import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):


    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "/Users/zhangjessie/PycharmProjects/pythonProject1/pointmass2.xml",2)
        self.action_space

    def step(self, a):
        ball_to_cylinder= self.get_body_com("torso") - self.get_body_com("cylinder")
        cylinder_to_target=self.get_body_com("targetz") - self.get_body_com("cylinder")
        ball_to_target = self.get_body_com("torso") - self.get_body_com("targetz")
        reward_dist =-np.linalg.norm(ball_to_cylinder) -3*np.linalg.norm(cylinder_to_target)
        reward_ball_to_target = np.linalg.norm(ball_to_target)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + 0.0*reward_ctrl
        scale=2
        y=np.random.randn(1,1,10)
        # if y>scale:
        #     scale=y
        # else:
        #     scale=

        #reward = -np.linalg.norm(ball_to_cylinder)
        #a=a+np.random.randn(len(a))*scale

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.2

    def reset_model(self,res=True):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )

        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        #print("qos1:")
        #print(qpos.shape())
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0



        if res==False:
            qpos=np.array([ 0,  3.37308747e-02 , 2.16267663e-01, -9.50301480e-02,1.67921949e-03 , 9.90627615e-01 , 6.59464278e-02 , 2.87708863e-02,-1.36928846e-01])
            qvel=np.array([ 0.00047619, -0.00046151,  0.00455408,  0.00434091 ,-0.00234884 ,-0.004199, 0 , 0  ])
        self.set_state(qpos, qvel)
       # print(self._get_obs())
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("torso") - self.get_body_com("targetz"),
            ]
        )


if __name__ == "__main__":
    env = PointEnv()
    #env=gym.make("Pusher")
    #max_a = env.action_space
    #while True:
    #    pass
    for j in range(10):
        env.reset()
        for i in range(1000):
            a = env.action_space.sample() #np.random.randn(2)
            env.step(a)
            env.render()
