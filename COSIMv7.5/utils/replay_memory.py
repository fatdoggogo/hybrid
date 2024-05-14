import random
import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def detach_and_convert_to_numpy(self, item):
        if isinstance(item, torch.Tensor):
            # 如果张量在 CUDA 上，先将其移动到 CPU 上
            if item.is_cuda:
                item = item.detach().cpu()
            return item.numpy()
        elif isinstance(item, list):
            arrays = [self.detach_and_convert_to_numpy(sub_item) for sub_item in item]
            combined_array = np.concatenate([arr.flatten() if isinstance(arr, np.ndarray) else np.array([arr]).flatten() for arr in arrays])
            return combined_array
        else:
            return item

    def stack_arrays(self, arrays):

        target_shape = next(arr.shape for arr in arrays if arr is not None)
        arrays = [arr if arr is not None else np.full(target_shape, np.nan) for arr in arrays]
        arrays_to_stack = [arr for arr in arrays if arr.size > 0]

        return np.vstack(arrays_to_stack)

    def push(self, state, action_c, action_d, weights, reward, next_state, done):  # action是一个包含两个张量的列表：[action_c, action_d]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
                                      self.detach_and_convert_to_numpy(state),
                                      self.detach_and_convert_to_numpy(action_c),
                                      self.detach_and_convert_to_numpy(action_d),
                                      self.detach_and_convert_to_numpy(weights).flatten(),
                                      self.detach_and_convert_to_numpy(reward),
                                      self.detach_and_convert_to_numpy(next_state),
                                      done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action_c, action_d, ws, rewards, next_states, dones = zip(*batch)
        state = self.stack_arrays(states)
        action_c = self.stack_arrays(action_c)
        action_d = self.stack_arrays(action_d)
        w = self.stack_arrays(ws)
        reward = self.stack_arrays(rewards)
        next_state = self.stack_arrays(next_states)
        done = np.array(dones).reshape(-1, 1)

        return state, action_c, action_d, w, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
