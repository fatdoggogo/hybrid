from env import Env
import RLBrain_DDPG


def run():
    env = Env(1, "rl_ddpg")
    myDdpg = RLBrain_DDPG.DDPG(actorNet=RLBrain_DDPG.Actor, criticNet=RLBrain_DDPG.Critic, env=env)
    myDdpg.learn()


if __name__ == '__main__':
    run()
