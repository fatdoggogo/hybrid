from env import Env
import RLBrain_DDPG1



def run():
    env = Env(1, "rl_ddpg")
    myDdpg = RLBrain_DDPG1.DDPG(actorNet=RLBrain_DDPG1.Actor, criticNet=RLBrain_DDPG1.Critic, env=env)
    myDdpg.learn()



if __name__ == '__main__':
    run()
