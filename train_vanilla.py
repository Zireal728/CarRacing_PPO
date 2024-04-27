"""
是最应该吃透的代码
"""

import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import DrawLine

# # 参数处理，参数最后被传递到args中
# parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
# parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
# parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
# parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
# parser.add_argument('--render', action='store_true', help='render the environment')
# parser.add_argument('--vis', action='store_true', help='use visdom')
# parser.add_argument(
#     '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
# args = parser.parse_args()

gamma = 0.99
action_repeat = 8
img_stack = 4
seed = 0
log_interval = 10
vis = True



# 设备配置
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    print("use cuda")
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

# 创建了一个numpy结构化类型对象，用于定义结构化数据
# 第一行表示由 args.img_stack 个 96*96构成的一个state
# 第二行表示动作，意味着每一个动作由三个浮点数构成
# rward
# s_, new state

transition = np.dtype([('s', np.float64, (img_stack, 96, 96)),
                       ('a', np.float64, (3,)),
                       ('a_logp', np.float64),
                       ('r', np.float64),
                       ('s_', np.float64, (img_stack, 96, 96))])

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        # (['human', 'rgb_array', 'state_pixels', 'single_rgb_array', 'single_state_pixels']).
        # self.env = gym.make('CarRacing-v2', new_step_api=True, render_mode = "human")
        self.env = gym.make('CarRacing-v2', new_step_api=True)
        # self.env.seed(args.seed)
        self.env.reset(seed = seed)
        # 设置环境的奖励阈值，判断是否完成任务或者评估性能
        self.reward_threshold = self.env.spec.reward_threshold


    # 重设环境到初始状态，所以最后返回的应该是一个状态
    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        # 将四个frames to a state
        self.stack = [img_gray] * img_stack  # four frames for decision
        # 强制类型转换
        return np.array(self.stack)


    # 执行动作并且返回新的状态，奖励等信息
    def step(self, action):
        total_reward = 0
        # 多次执行同一个动作，这是一种常见的技巧，用于降低决策频率，提高训练效率
        for i in range(action_repeat):
            img_rgb, reward, die, _, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty 对于草地行驶给予惩罚
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            # 对于当前的episode，判断是不是结束
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == img_stack
        return np.array(self.stack), total_reward, done, die

    # 根据参数配置渲染环境
    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

# 使用卷积神经网络处理输入，并且使用两个输出，分别输出：预测的动作，和当前的价值（暂时不明确是Gt还是reward)
class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """
    # 用一个多层CNN处理数据
    # 然后分成两个输出
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        # self.v = nn.Sequential(...): 定义了一个小型全连接网络，
        # 用于从CNN提取的特征中预测当前状态的值（V）。
        # 进而估计Critic部分的价值函数
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        # 用于动作预测部分的进一步提取
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        # Beta分布，用于预测动作，可以模拟一大堆分布，包括单峰、增长、下降和均匀分布等
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        # 在PyTorch中，self.apply(fn)方法用于递归地将函数fn应用到模块本身以及它的每一个子模块（即网络中的每一层）。具体到self.apply(self._weights_init)这行代码，它的作用是递归地将_weights_init函数应用于神经网络的所有层。这意味着对网络中每一个层的权重执行自定义的初始化过程。
        # 其中 _weights_init在下面有定义
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    # 定义了网络的连接方式， 最后返回的是动作和奖励
    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    """
    Agent for training
    """
    # max_grad_norm：梯度裁剪的最大范数，用于防止梯度爆炸。
    # clip_param：PPO的裁剪参数ε，用于限制策略更新的幅度。
    # ppo_epoch：每次使用收集的数据进行训练的轮数。
    # buffer_capacity和batch_size：分别代表经验回放缓冲区的容量和每个训练批次的大小。
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        # 训练步数
        # 训练步数
        self.training_step = 0
        self.net = Net().double().to(device)
        # 存储经验数据
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        # 定义优化器
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    # pi函数 pi(state)
    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        #
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        # 使用Beta分布来采样动作
        dist = Beta(alpha, beta)
        action = dist.sample()
        # 计算采样动作的对数概率 （也就是把数转变为一个概率）
        a_logp = dist.log_prob(action).sum(dim=1)

        # 移除张量中所有长度为1的维度
        action = action.squeeze().cpu().numpy()
        # 选择这个动作的对数概率
        a_logp = a_logp.item()
        # 返回一个动作和一个对数概率
        return action, a_logp

    def save_param(self, current_epoch):
        torch.save(self.net.state_dict(), f'param/ppo_net_params_{current_epoch}.pkl')

    #
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    agent = Agent()
    env = Env()
    if vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(100000):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            # if args.render:
            #     env.render()
            #
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % log_interval == 0:
            if vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param(i_ep)
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
