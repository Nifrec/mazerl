"""
Function (and helper functions) used to launch the training of a TD3 agent
using Asynchronous Methods through muliprocessing.

Author: Lulof Pirée
"""

def create_processes():
    gradientQueue = multiprocessing.Queue()
        processes = []

        num_agents = len(self.__agents)
        i = 0
        for _ in range(self.__num_processes):
            agent = self.__agents[i]
            i = (i + 1) % num_agents

            worker = AgentWorker(self.__hyperparameters, agent, 
                    self.__logger, gradientQueue)
            #p = multiprocessing.Process(target=worker.run)
            p = threading.Thread(target=worker.run)
            processes.append(p)
            p.start()