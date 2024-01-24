from env import Env
import RLBrain_Hybrid



def run():
    env = Env(1, "rl_hybrid")
    myHybrid = RLBrain_Hybrid.Hybrid(actor=RLBrain_Hybrid.ActorNet, critic=RLBrain_Hybrid.CriticNet, env=env)
    myHybrid.learn(total_timesteps=env.episodes)



if __name__ == '__main__':
    run()
