from env import Env
import RLBrain_CAPQL


if __name__ == '__main__':
    env = Env(1, "rl_capql")
    capql = RLBrain_CAPQL.CAPQL(env=env)
    capql.run()
